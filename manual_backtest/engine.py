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


DEFAULT_CONFIG = {
    "sector_rps_min": 40.0,
    "w_l2": 0.50,
    "w_stock_rps": 0.25,
    "w_sector_rps": 0.00,
    "w_qlib": 0.25,
}


class ManualBacktester:
    """人工回测编排器 — 截面日期 L1→L4 管道 + 人工标记后逐只回测."""

    def __init__(self, config: dict | None = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.sector_rps_min = cfg["sector_rps_min"]
        self.w_l2 = cfg["w_l2"]
        self.w_stock_rps = cfg["w_stock_rps"]
        self.w_sector_rps = cfg["w_sector_rps"]
        self.w_qlib = cfg["w_qlib"]
        self.l4_df: pd.DataFrame | None = None
        self.marked: pd.DataFrame | None = None
        self.trades_df: pd.DataFrame | None = None
        self._current_date: str = ""

    # ── L1→L4 管道 ──────────────────────────────────────────

    def run_pipeline(self, date: str) -> pd.DataFrame:
        """对指定截面日期运行 L1→L2→L3→L4 完整评分管道."""
        self._current_date = date
        target_date = pd.Timestamp(date).date()
        regime = detect_regime(date)

        # Step 1: 快速预扫 — 扫描全部信号文件找当日新出现的买点
        candidates = []
        files = sorted(SIGNALS_DIR.glob("*.parquet"))
        if not files:
            self.l4_df = pd.DataFrame()
            return self.l4_df
        buy_cols = [c for c in BUY_COLS if c in pd.read_parquet(files[0]).columns]

        for fpath in files:
            code = fpath.stem
            try:
                sig_df = pd.read_parquet(fpath)
            except Exception:
                continue
            if len(sig_df) == 0:
                continue
            sig_df["_d"] = pd.to_datetime(sig_df["dt"]).dt.date
            mask = sig_df["_d"] == target_date
            if mask.sum() == 0:
                continue
            for idx in mask[mask].index:
                if idx == 0:
                    continue
                for col in buy_cols:
                    old_v = str(sig_df.at[idx - 1, col])
                    new_v = str(sig_df.at[idx, col])
                    if old_v == new_v or old_v == "nan":
                        continue
                    if new_v in ("", "nan", "None", "0"):
                        continue
                    candidates.append({
                        "code": code,
                        "signal_label": new_v,
                        "date": str(sig_df.at[idx, "dt"]),
                    })
                    break

        if not candidates:
            self.l4_df = pd.DataFrame()
            return self.l4_df

        # Step 2: L2 EntryFilter 入场打分
        l2_rows = []
        for cand in candidates:
            code = cand["code"]
            try:
                daily = load_daily(code)
                sig_df = load_signals(code)
                if daily is None or sig_df is None:
                    continue
                entry_filter = EntryFilter(code, daily, sig_df, regime)
                result = entry_filter.filter(cand)
                l2_rows.append({
                    "code": code,
                    "buy_type": result.buy_type,
                    "signal_date": cand["date"],
                    "total_score": round(result.total_score, 1),
                    "regime": regime,
                    "passed": result.passed,
                    "reject_reason": result.reject_reason,
                })
            except Exception:
                continue

        if not l2_rows:
            self.l4_df = pd.DataFrame()
            return self.l4_df

        l2_df = pd.DataFrame(l2_rows)

        # Step 3: L3 质量过滤
        l3_filter = L3Filter(regime)
        l3_df = l3_filter.filter_batch(l2_df)

        # Step 4: L4 排名
        ranker = L4Ranker(
            sector_rps_min=self.sector_rps_min,
            w_l2=self.w_l2, w_stock_rps=self.w_stock_rps,
            w_sector_rps=self.w_sector_rps, w_qlib=self.w_qlib,
        )
        l4_df = ranker.rank(l3_df)

        # Step 5: 附加行业名称
        industry_map = _get_industry_map()
        if not industry_map.empty and "industry" in industry_map.columns:
            ind_lookup = industry_map.set_index("code")["industry"].to_dict()
            l4_df["sector"] = l4_df["code"].map(ind_lookup).fillna("")
        else:
            l4_df["sector"] = ""
        l4_df["regime"] = regime
        l4_df["signal_date"] = date

        self.l4_df = l4_df.reset_index(drop=True)
        return self.l4_df

    # ── CSV 输出 ────────────────────────────────────────────

    def export_for_marking(self, out_dir: str | None = None) -> Path:
        """导出 L4 报告 CSV，含 selected=0 空列供人工标记."""
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
