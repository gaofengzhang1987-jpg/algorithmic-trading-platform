"""ManualBacktester — L1→L4 管道编排 + 人工标记回测."""
from pathlib import Path

import pandas as pd

from core.constants import MAX_HOLD_DAYS, COMMISSION, SIGNALS_DIR, BUY_COLS
from core.data import load_daily, load_signals, get_next_trading_day, get_price_at_date
from core.signal_detector import detect_all_changes
from core.structure_cache import load_structure_for_code
from entry_filter import EntryFilter
from l3_filter import L3Filter
from l4_ranker import L4Ranker
from backtest.exit_engine import ExitEngine
from manual_backtest.report import export_l4_csv, export_trades_csv


_OUT_BASE = Path(__file__).parent.parent / "tmp_out" / "manual_backtest"

# ── regime 检测（复用上证指数 MA20/MA60 排列） ────────────────

_INDEX_CODE = "000001"
_INDEX_DF = None


def _load_index() -> pd.DataFrame:
    global _INDEX_DF
    if _INDEX_DF is None:
        df = load_daily(_INDEX_CODE)
        if df is None:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        _INDEX_DF = df.sort_values("date").reset_index(drop=True)
    return _INDEX_DF


def detect_regime(signal_date_str: str) -> str:
    """基于上证指数 MA20/MA60 多头空头排列判断 regime."""
    idx = _load_index()
    if idx.empty:
        return "CHOP"
    sig_dt = pd.Timestamp(signal_date_str)
    window = idx[idx["date"] <= sig_dt].tail(60)
    if len(window) < 40:
        return "CHOP"
    c = window["close"]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    if pd.isna(ma60):
        return "CHOP"
    if c.iloc[-1] > ma20 > ma60:
        return "BULL"
    if c.iloc[-1] < ma20 < ma60:
        return "BEAR"
    return "CHOP"


# ── 行业名称映射 ────────────────────────────────────────────

_INDUSTRY_MAP_CACHE = None


def _get_industry_map() -> pd.DataFrame:
    global _INDUSTRY_MAP_CACHE
    if _INDUSTRY_MAP_CACHE is None:
        p = Path(__file__).parent.parent / "data" / "industry_classification.parquet"
        if p.exists():
            _INDUSTRY_MAP_CACHE = pd.read_parquet(p)
        else:
            _INDUSTRY_MAP_CACHE = pd.DataFrame(columns=["code", "industry"])
    return _INDUSTRY_MAP_CACHE


_OUT_BASE = Path(__file__).parent.parent / "tmp_out" / "manual_backtest"


DEFAULT_CONFIG = {}


class ManualBacktester:
    """人工回测编排器 — 截面日期采用最新的信号数据，走全量 zone1-zone4 漏斗。"""

    def __init__(self, config: dict | None = None):
        _ = config or {}  # 当前无配置项，保留兼容
        self.l4_df: pd.DataFrame | None = None
        self.marked: pd.DataFrame | None = None
        self.trades_df: pd.DataFrame | None = None
        self._current_date: str = ""

    # ── L1→L4 管道 (与 run_zones.py 一致) ──────────────────────

    def run_pipeline(self, date: str = "") -> pd.DataFrame:
        """运行全量 L1→L2→L3→L4 漏斗。

        date 参数用于输出目录命名和 signal_date 列，不影响检测逻辑（zone1 始终基于最新信号数据）。
        """
        if not date:
            date = pd.Timestamp.now().strftime("%Y-%m-%d")
        self._current_date = date

        # ── L1: zone1_deposition ──────────────────────────────
        print(f"  [L1] zone1_deposition (lookback=20)...", flush=True)
        df1 = zone1_run(lookback=20)
        if df1.empty:
            print(f"  [L1] 无候选", flush=True)
            self.l4_df = pd.DataFrame()
            return self.l4_df
        print(f"  [L1] {len(df1)} 只", flush=True)

        # ── L1→L2 过渡: regime + B+ ───────────────────────────
        regime, score, dims = regime_detector.detect()
        print(f"  [Regime] {regime} (BullScore={score:.0f}/100)", flush=True)

        bplus_codes = set()
        resonance_count = 0
        for _, row in df1.iterrows():
            code = row["代码"]
            label = row["买点类型"]
            bt = "一买" if "一买" in label else ("二买" if "二买" in label else "三买")
            if bt != "一买" and verify_buy_type(code, bt):
                bplus_codes.add((code, bt))
            elif check_resonance(code, bt, signal_date=row.get("最新日期")):
                bplus_codes.add((code, bt))
                resonance_count += 1
        print(f"  [B+] {len(bplus_codes)} 只 (结构{len(bplus_codes)-resonance_count} + 共振{resonance_count})", flush=True)

        # ── L2: zone2_regime ──────────────────────────────────
        print(f"  [L2] zone2_regime...", flush=True)
        df2 = zone2_run(df1, regime=regime, bplus_codes=bplus_codes)
        print(f"  [L2] {len(df2)} 只", flush=True)

        # ── L3: zone3_regime ──────────────────────────────────
        print(f"  [L3] zone3_regime...", flush=True)
        df3 = zone3_run(df2, regime=regime)
        print(f"  [L3] {len(df3)} 只", flush=True)

        # ── L4: zone4_regime ──────────────────────────────────
        print(f"  [L4] zone4_regime...", flush=True)
        df4 = zone4_run(df3, top_n=9999)
        print(f"  [L4] {len(df4)} 只", flush=True)

        # ── 转换到我们的 L4 格式 ──────────────────────────────
        l4 = pd.DataFrame()
        if not df4.empty:
            l4["code"] = df4["代码"].astype(str)
            l4["buy_type"] = df4.get("买点类型", df4.get("buy_type", "")).astype(str)
            l4["signal_date"] = date
            l4["regime"] = regime
            l4["composite"] = df4.get("L4_综合得分", df4.get("composite", 0.0))
            l4["global_rank"] = range(1, len(l4) + 1)
            l4["zone_rank"] = 0
            l4["n_l2"] = df4.get("L2_norm", 0.0) if "L2_norm" in df4.columns else 0.0
            l4["stock_rps"] = df4.get("stock_rps", 0.0) if "stock_rps" in df4.columns else 0.0
            l4["sector_rps"] = df4.get("sector_rps", 0.0) if "sector_rps" in df4.columns else 0.0
            l4["qlib_score"] = 0.5
            l4["sector"] = df4.get("行业", df4.get("sector", "")).fillna("").astype(str) if "行业" in df4.columns or "sector" in df4.columns else ""
            l4["total_score"] = df4.get("L2_综合得分", df4.get("total_score", 0.0))
            l4["passed"] = True

        self.l4_df = l4.reset_index(drop=True)
        return self.l4_df

    # ── CSV 输出 ────────────────────────────────────────────

    def export_for_marking(self, out_dir: str | None = None) -> Path:
        if self.l4_df is None or self.l4_df.empty:
            raise ValueError("L4 数据为空，请先 run_pipeline")
        d = Path(out_dir) if out_dir else _OUT_BASE / self._current_date
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / f"l4_{self._current_date}.csv"
        return export_l4_csv(self.l4_df, out_path)

    def load_marked(self, path: str | Path) -> pd.DataFrame:
        """读取已标记 CSV (selected=1 的行)."""
        path = Path(path)
        if path.is_dir():
            dfs = []
            for f in sorted(path.glob("*_marked.csv")):
                dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
            if not dfs:
                raise FileNotFoundError(f"目录 {path} 下无 _marked.csv 文件")
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(path, encoding="utf-8-sig")
        self.marked = df[df["selected"] == 1].reset_index(drop=True)
        return self.marked

    # ── 回测执行 ────────────────────────────────────────────

    def backtest_selected(self) -> pd.DataFrame:
        """对标记股票逐只回测，返回 trades_df."""
        if self.marked is None or self.marked.empty:
            raise ValueError("未加载标记数据，请先 load_marked")
        trades = []
        for _, row in self.marked.iterrows():
            code = str(row["code"])
            buy_type = str(row.get("buy_type", ""))
            signal_date = str(row.get("signal_date", self._current_date))
            l4_rank = int(row.get("global_rank", 0))
            composite = float(row.get("composite", 0))
            regime = str(row.get("regime", "CHOP"))
            try:
                daily = load_daily(code)
                sig_df = load_signals(code)
                if daily is None or sig_df is None:
                    continue
                struct_df = load_structure_for_code(code)
                changes = detect_all_changes(sig_df)
                buy_events = [c for c in changes if c["type"] == "buy"]
                sell_events = [c for c in changes if c["type"] == "sell"]
                target_d = pd.Timestamp(signal_date).date()
                # 匹配当日之前的最近买点
                matched = None
                bt_keyword = {"一买": "一买", "二买": "二买", "三买": "三买"}.get(buy_type, "一买")
                for be in reversed(buy_events):
                    be_d = pd.Timestamp(be["date"]).date()
                    if be_d <= target_d and bt_keyword in be.get("signal_label", ""):
                        matched = be
                        break
                if matched is None:
                    for be in reversed(buy_events):
                        be_d = pd.Timestamp(be["date"]).date()
                        if be_d <= target_d:
                            matched = be
                            break
                if matched is None:
                    continue
                daily_sorted = daily.sort_values("date").reset_index(drop=True)
                entry_date = get_next_trading_day(matched["date"], daily_sorted)
                if entry_date is None:
                    continue
                entry_price = get_price_at_date(entry_date, daily_sorted)
                if entry_price is None or entry_price <= 0:
                    continue
                engine = ExitEngine(code, entry_price, entry_date,
                                    matched["signal_label"], struct_df,
                                    trajectory_log=True)
                sell_exit_target = None
                for sell in sell_events:
                    s_d = pd.Timestamp(sell["date"])
                    if s_d > entry_date:
                        cand = get_next_trading_day(sell["date"], daily_sorted)
                        if cand is not None:
                            sell_exit_target = cand
                        break
                entry_idx = daily_sorted[daily_sorted["date"] == entry_date].index
                if entry_idx.empty:
                    continue
                entry_idx = entry_idx[0]
                end_idx = min(entry_idx + MAX_HOLD_DAYS + 1, len(daily_sorted))
                window = daily_sorted.iloc[entry_idx + 1:end_idx]
                if len(window) == 0:
                    continue
                exit_reason = ""
                exit_price_val = None
                exit_date = None
                weighted_ret = None
                trajectory = []
                for bar_i in range(len(window)):
                    bar = window.iloc[bar_i]
                    bar_date = bar["date"]
                    bar_close = bar["close"]
                    bar_high = bar["high"]
                    result = engine.process_bar(bar_date, bar_close, bar_high,
                                                sell_exit_target)
                    if result.exit:
                        exit_price_val = bar_close
                        if result.exit_reason == "卖点":
                            exit_price_val = get_price_at_date(bar_date, daily_sorted) or bar_close
                        weighted_ret = engine.compute_weighted_return(
                            entry_price, exit_price_val)
                        exit_date = bar_date
                        exit_reason = result.exit_reason
                        trajectory = result.trajectory if result.trajectory else []
                        break
                if exit_price_val is None:
                    last_bar = window.iloc[-1]
                    exit_date = last_bar["date"]
                    exit_price_val = float(last_bar["open"])
                    exit_reason = "到期"
                    weighted_ret = engine.compute_weighted_return(
                        entry_price, exit_price_val)
                    trajectory = engine._trajectory if engine.trajectory_log else []
                if weighted_ret is None or exit_price_val is None:
                    continue
                net_return = weighted_ret - 2 * COMMISSION
                hold_days = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
                trades.append({
                    "code": code,
                    "buy_type": buy_type,
                    "signal_date": signal_date,
                    "entry_date": str(pd.Timestamp(entry_date).date()),
                    "exit_date": str(pd.Timestamp(exit_date).date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price_val, 2),
                    "return_pct": round(net_return * 100, 2),
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                    "l4_rank": l4_rank,
                    "composite": round(composite, 4),
                    "regime": regime,
                    "trajectory": trajectory,
                })
            except Exception:
                continue
        self.trades_df = pd.DataFrame(trades)
        return self.trades_df

    def backtest_auto_top_n(self, top_n: int = 50) -> pd.DataFrame:
        """自动 top-N 回测（用于对比基准）."""
        if self.l4_df is None or self.l4_df.empty:
            raise ValueError("L4 数据为空，请先 run_pipeline")
        self.marked = self.l4_df.head(top_n).copy()
        self.marked["selected"] = 1
        return self.backtest_selected()
