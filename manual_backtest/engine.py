"""ManualBacktester — L1→L4 管道编排 + 人工标记回测.

复用 zone1_deposition → zone2_regime → zone3_regime → zone4_regime 全漏斗，
与 l1_update.py / run_zones.py 的数据更新管线完全一致。
"""
import json
import time
from pathlib import Path

import pandas as pd

from core.constants import MAX_HOLD_DAYS, COMMISSION, SIGNALS_DIR
from core.data import load_daily, load_signals, get_next_trading_day, get_price_at_date
from core.signal_detector import detect_all_changes
from core.structure_cache import load_structure_for_code
from backtest.exit_engine import ExitEngine
from manual_backtest.analyzer import ManualAnalyzer
from manual_backtest.report import export_l4_csv, export_trades_csv, print_summary

from zone1_deposition import run as zone1_run
from zone2_regime import run as zone2_run
from zone3_regime import run as zone3_run
from zone4_regime import run as zone4_run
import regime_detector
from verify_buy_type import verify_buy_type, check_resonance

_OUT_BASE = Path(__file__).parent.parent / "tmp_out" / "manual_backtest"
_BASE_DIR = Path(__file__).parent.parent
_STOCK_NAME_CACHE: dict[str, str] | None = None


def _stock_name_map() -> dict[str, str]:
    """加载本地股票代码->名称映射，避免回测依赖网络请求。"""
    global _STOCK_NAME_CACHE
    if _STOCK_NAME_CACHE is None:
        path = _BASE_DIR / "data" / "industry_classification.parquet"
        try:
            df = pd.read_parquet(path)
            _STOCK_NAME_CACHE = dict(zip(df["code"].astype(str).str.zfill(6),
                                        df["name"].astype(str)))
        except Exception:
            _STOCK_NAME_CACHE = {}
    return _STOCK_NAME_CACHE

DEFAULT_CONFIG = {}

class ManualBacktester:
    """人工回测编排器 — 截面日期采用最新信号数据，走全量 zone1-zone4 漏斗。"""

    def __init__(self, config: dict | None = None):
        _ = config or {}
        self.l4_df: pd.DataFrame | None = None
        self.marked: pd.DataFrame | None = None
        self.trades_df: pd.DataFrame | None = None
        self._current_date: str = ""

    # ── L1→L4 管道 (与 run_zones.py 一致) ──────────────────────

    def run_pipeline(self, date: str = "") -> pd.DataFrame:
        """运行全量 L1→L2→L3→L4 漏斗。date 仅用于输出目录命名。"""
        if not date:
            date = pd.Timestamp.now().strftime("%Y-%m-%d")
        self._current_date = date

        # L1
        print(f"  [L1] zone1_deposition...", flush=True)
        df1 = zone1_run(lookback=20)
        if df1.empty:
            print(f"  [L1] 无候选", flush=True)
            self.l4_df = pd.DataFrame()
            return self.l4_df
        print(f"  [L1] {len(df1)} 只", flush=True)

        # Regime + B+
        result = regime_detector.detect(); regime = result[0] if isinstance(result, tuple) else result
        print(f"  [Regime] {regime}", flush=True)

        bplus_codes = set()
        bplus_verify_count = 0
        resonance_count = 0
        for _, row in df1.iterrows():
            code = row["代码"]
            label = row["买点类型"]
            bt = "一买" if "一买" in label else ("二买" if "二买" in label else "三买")
            sig_date = row.get("最新日期")
            if bt == "一买":
                continue
            if verify_buy_type(code, bt):
                bplus_verify_count += 1  # 加分不豁免
            if check_resonance(code, bt, signal_date=sig_date):
                resonance_count += 1
        print(f"  [B+] {bplus_verify_count}只(加分) 标签共振{resonance_count}只", flush=True)

        # L2
        print(f"  [L2] zone2_regime...", flush=True)
        # 历史截面适配：临时覆写 pd.Timestamp.now 为截面日期，绕过新鲜度闸门
        import pandas as _pd
        _real_now = _pd.Timestamp.now
        _fake_date = _pd.Timestamp(date)
        _pd.Timestamp.now = staticmethod(lambda tz=None: _fake_date)
        try:
            df2 = zone2_run(df1, regime=regime, bplus_codes=bplus_codes)
        finally:
            _pd.Timestamp.now = _real_now
        print(f"  [L2] {len(df2)} 只", flush=True)

        # L3
        print(f"  [L3] zone3_regime...", flush=True)
        df3 = zone3_run(df2, regime=regime)
        print(f"  [L3] {len(df3)} 只", flush=True)

        # L4
        print(f"  [L4] zone4_regime...", flush=True)
        df4 = zone4_run(df3, top_n=9999, skip_fundamental=True, signal_date=date)
        print(f"  [L4] {len(df4)} 只", flush=True)

        # 转换到我们的 L4 格式
        l4 = pd.DataFrame()
        if not df4.empty:
            l4["code"] = df4["code"].astype(str).str.zfill(6)
            l4["name"] = l4["code"].map(_stock_name_map()).fillna("")
            l4["buy_type"] = df4["买点类型"].astype(str)
            l4["signal_date"] = date
            l4["regime"] = regime
            l4["composite"] = df4.get("composite", 0.0)
            l4["global_rank"] = range(1, len(l4) + 1)
            l4["zone_rank"] = df4.get("zone_rank", 0)
            l4["n_l2"] = df4.get("n_l2", 0.0)
            l4["stock_rps"] = df4.get("stock_rps", 0.0)
            l4["sector_rps"] = df4.get("sector_rps", 0.0)
            l4["qlib_score"] = df4.get("qlib_score", 0.5)
            l4["sector"] = df4["行业"].fillna("").astype(str) if "行业" in df4.columns else ""
            l4["total_score"] = df4.get("L2_综合得分", 0.0)
            l4["passed"] = True
            l4["非买点转买点"] = (
                df4["非买点转买点"].fillna(False).astype(bool)
                if "非买点转买点" in df4.columns else False
            )
            l4["当天非买转买"] = (
                df4["当天非买转买"].fillna(False).astype(bool)
                if "当天非买转买" in df4.columns else False
            )

        # top 100 + 三级联立豁免（2026-08-09）
        if l4 is not None and len(l4) > 0 and 'buy_type' in l4.columns:
            top100 = l4.head(100)
            exempt = l4[l4['buy_type'].str.contains('结构联立', na=False)]
            l4 = pd.concat([top100, exempt]).drop_duplicates(subset=['code']).reset_index(drop=True)
            l4['global_rank'] = range(1, len(l4) + 1)
        elif l4 is None:
            l4 = pd.DataFrame()

        self.l4_df = l4.reset_index(drop=True)
        return self.l4_df

    # ── CSV 输出 ────────────────────────────────────────────

    def export_for_marking(self, out_dir: str | None = None) -> Path:
        if self.l4_df is None or self.l4_df.empty:
            raise ValueError("L4 数据为空，请先 run_pipeline")
        d = Path(out_dir) if out_dir else _OUT_BASE / self._current_date
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / f"l4_{self._current_date}.csv"
        try:
            from selection_engine import load_rules, select
            rec = select(self.l4_df, load_rules())
            rec.to_csv(d / f"recommend_{self._current_date}.csv",
                       index=False, encoding="utf-8-sig")
        except Exception:
            rec = None
        return export_l4_csv(self.l4_df, out_path, recommendations=rec)

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
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)
        self.marked = df[df["selected"] == 1].reset_index(drop=True)
        return self.marked

    def backtest_selected(self) -> pd.DataFrame:
        """对标记股票逐只回测."""
        if self.marked is None or self.marked.empty:
            raise ValueError("未加载标记数据，请先 load_marked")
        trades = []
        for i, (_, row) in enumerate(self.marked.iterrows()):
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
                matched = None
                bt_keyword = next((k for k in ("一买", "二买", "三买") if k in buy_type), "一买")
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
                entry_date = get_next_trading_day(signal_date, daily_sorted)
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
                for bi in range(len(window)):
                    bar = window.iloc[bi]
                    bar_date = bar["date"]
                    bar_close = bar["close"]
                    bar_high = bar["high"]
                    result = engine.process_bar(bar_date, bar_close, bar_high, sell_exit_target)
                    if result.exit:
                        exit_price_val = bar_close
                        if result.exit_reason == "卖点":
                            exit_price_val = get_price_at_date(bar_date, daily_sorted) or bar_close
                        weighted_ret = engine.compute_weighted_return(entry_price, exit_price_val)
                        exit_date = bar_date
                        exit_reason = result.exit_reason
                        trajectory = result.trajectory or []
                        break
                if exit_price_val is None:
                    last_bar = window.iloc[-1]
                    exit_date = last_bar["date"]
                    exit_price_val = float(last_bar["open"])
                    exit_reason = "到期"
                    weighted_ret = engine.compute_weighted_return(entry_price, exit_price_val)
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
                    "非买点转买点": str(row.get("非买点转买点", False)).strip().lower() == "true",
                    "当天非买转买": str(row.get("当天非买转买", False)).strip().lower() == "true",
                    "trajectory": trajectory,
                })
            except Exception:
                pass
        self.trades_df = pd.DataFrame(trades)
        return self.trades_df

    def backtest_auto_top_n(self, top_n: int = 50) -> pd.DataFrame:
        """自动 top-N 回测（用于对比）."""
        if self.l4_df is None or self.l4_df.empty:
            raise ValueError("L4 数据为空，请先 run_pipeline")
        self.marked = self.l4_df.head(top_n).copy()
        self.marked["selected"] = 1
        return self.backtest_selected()

    def backtest_all_l4(self, l4_dir=None, out_dir=None, limit: int = 0,
                        force: bool = False) -> pd.DataFrame:
        """对全部已完成 L4 截面做出场回测，所有行视为人工选中。

        与 backtest_selected 共用同一套回测逻辑，仅把 selected 全部置 1。
        """
        l4_dir = Path(l4_dir) if l4_dir else _BASE_DIR / "tmp_out"
        out_dir = Path(out_dir) if out_dir else l4_dir / "exit_backtest"
        out_dir.mkdir(parents=True, exist_ok=True)
        progress = out_dir / "progress.txt"
        l4_files = sorted(l4_dir.glob("backtest_*/l4_*.csv"))

        total_rows = 0
        total_trades = 0
        t0 = time.time()

        for fi, f in enumerate(l4_files, 1):
            df = pd.read_csv(f, encoding="utf-8-sig")
            if df.empty:
                continue
            section_date = str(df["signal_date"].iloc[0])
            per_section_out = out_dir / f"trades_{section_date}.csv"

            if per_section_out.exists() and not force:
                n = len(pd.read_csv(per_section_out, encoding="utf-8-sig"))
                total_trades += n
                msg = f"[{fi}/{len(l4_files)}] {section_date} SKIP {n} trades {time.time()-t0:.0f}s"
                progress.write_text(msg + "\n")
                print(msg, flush=True)
                continue

            if limit and total_rows >= limit:
                break
            take = df
            if limit:
                take = df.head(limit - total_rows)

            take = take.copy()
            take["selected"] = 1
            if "code" in take.columns:
                take["code"] = take["code"].astype(str).str.zfill(6)
            self.marked = take.reset_index(drop=True)
            self._current_date = section_date
            trades = self.backtest_selected()

            total_rows += len(take)
            total_trades += len(trades)
            if not trades.empty:
                trades = trades.copy()
                trades["section_date"] = section_date
                export_trades_csv(trades, per_section_out)

            msg = (f"[{fi}/{len(l4_files)}] {section_date} rows={len(take)} "
                   f"trades={len(trades)} skip={len(take)-len(trades)} "
                   f"{time.time()-t0:.0f}s")
            progress.write_text(msg + "\n")
            print(msg, flush=True)

        combined = []
        for pf in sorted(out_dir.glob("trades_*.csv")):
            combined.append(pd.read_csv(pf, encoding="utf-8-sig"))
        all_trades = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

        trades_path = out_dir / "trades_all.csv"
        export_trades_csv(all_trades, trades_path)
        if all_trades.empty:
            print("NO TRADES", flush=True)
            return all_trades

        stats = ManualAnalyzer(all_trades).analyze()
        with open(out_dir / "stats_all.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n合并交易: {len(all_trades)} 笔 -> {trades_path}", flush=True)
        print_summary(stats)
        return all_trades
