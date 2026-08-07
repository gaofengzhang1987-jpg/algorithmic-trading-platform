#!/usr/bin/env python3
"""烟雾测试 — 指定信号节点 → L1-L4 完整管道 → top100 回测。

Usage:
    python3 backtest/smoke_test.py 2024-02-05 2024-09-30 2025-04-15
    python3 backtest/smoke_test.py --top 50 --regime CHOP 2024-08-02
"""
import sys, os, argparse, time
from pathlib import Path

import pandas as pd
import numpy as np

from core.constants import (
    BASE_DIR, SIGNALS_DIR, BUY_COLS, MAX_HOLD_DAYS, COMMISSION,
)
from core.signal_parser import parse_signal
from core.data import load_daily, load_signals, get_next_trading_day, get_price_at_date
from core.signal_detector import detect_all_changes
from core.structure_cache import load_structure_for_code
from entry_filter import EntryFilter
from l3_filter import L3Filter
from l4_ranker import L4Ranker
try:
    from qlib_ml import QlibPredictor
except ImportError:
    QlibPredictor = None
from backtest.exit_engine import ExitEngine

PROGRESS = Path(__file__).parent.parent / "tmp_out" / "smoke_test_progress.txt"


def log(msg: str):
    print(msg, flush=True)
    PROGRESS.parent.mkdir(exist_ok=True)
    PROGRESS.write_text(msg)


# ── 自动 regime 检测 ──────────────────────────────────────────────

_INDEX_CODE = "000001"  # 上证指数 (fallback)
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
    """基于上证指数的 MA20/MA60 多头空头排列自动判断 regime。"""
    idx = _load_index()
    if idx.empty:
        return "CHOP"
    sig_dt = pd.Timestamp(signal_date_str)
    window = idx[idx["date"] <= sig_dt].tail(60)
    if len(window) < 40:
        return "CHOP"
    c = window["close"]
    ma20, ma60 = c.rolling(20).mean().iloc[-1], c.rolling(60).mean().iloc[-1]
    if pd.isna(ma60):
        return "CHOP"
    if c.iloc[-1] > ma20 > ma60:
        return "BULL"
    if c.iloc[-1] < ma20 < ma60:
        return "BEAR"
    return "CHOP"


# ── 阶段 1: 快速预扫 ──────────────────────────────────────────────

def fast_scan_buys(target_date_str: str) -> list:
    target_date = pd.Timestamp(target_date_str).date()
    candidates = []
    files = sorted(SIGNALS_DIR.glob("*.parquet"))
    total = len(files)

    first = pd.read_parquet(files[0])
    buy_cols = [c for c in BUY_COLS if c in first.columns]
    if not buy_cols:
        log("ERROR: no BUY_COLS available")
        return []

    for i, fpath in enumerate(files):
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
            if (i + 1) % 1000 == 0:
                log(f"  预扫 {i+1}/{total}, 候选 {len(candidates)}")
            continue

        for idx in mask[mask].index:
            if idx == 0:
                continue
            for col in buy_cols:
                if col not in sig_df.columns:
                    continue
                old_v = str(sig_df.at[idx - 1, col])
                new_v = str(sig_df.at[idx, col])
                if old_v == new_v or old_v == "nan":
                    continue
                if new_v in ("", "nan", "None", "0"):
                    continue
                parsed = parse_signal(new_v)
                bt = parsed.get("v1", "")
                if bt not in ("一买", "二买", "三买"):
                    continue
                col_suffix = col.split('_')[2][:8] if '_' in col and len(col.split('_')) > 2 else col[:8]
                sig_label = f"{new_v}({col_suffix})"
                candidates.append((code, {
                    "date": target_date_str,
                    "signal_label": sig_label,
                    "type": "buy",
                    "col": col,
                }))
        if (i + 1) % 500 == 0:
            log(f"  预扫 {i+1}/{total}, 候选 {len(candidates)}")

    log(f"  预扫完成: {total} 只 → {len(candidates)} 只当天有买点")
    return candidates


# ── 阶段 2: L2 EntryFilter ────────────────────────────────────────

def run_l2(candidates: list, regime: str = "CHOP") -> pd.DataFrame:
    results = []
    total = len(candidates)

    for i, (code, buy_event) in enumerate(candidates):
        try:
            daily_df = load_daily(code)
            if daily_df is None:
                continue
            sig_df = load_signals(code)
            if sig_df is None:
                continue
            ef = EntryFilter(code, daily_df, sig_df, regime=regime)
            fr = ef.filter(buy_event, regime=regime)
            results.append({
                "code": code,
                "buy_type": fr.buy_type,
                "signal_date": buy_event["date"],
                "total_score": round(fr.total_score, 2),
                "passed": fr.passed,
                "reject_reason": fr.reject_reason,
            })
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            passed = sum(1 for r in results if r["passed"])
            log(f"  L2 {i+1}/{total}, 通过 {passed}/{len(results)}")

    df = pd.DataFrame(results)
    if df.empty:
        log("  L2: 无候选通过")
        return df
    passed = df["passed"].sum()
    log(f"  L2 完成 ({regime}): {len(df)} 候选, {passed} 通过")
    return df


# ── 阶段 3/4: L3 + L4 ─────────────────────────────────────────────

def run_l34(l2_df: pd.DataFrame, regime: str = "CHOP") -> pd.DataFrame:
    if l2_df.empty:
        return l2_df

    l3 = L3Filter(regime=regime)
    l3.thr["freshness_days"] = 99999  # 历史回测：绕过信号新鲜度检查
    l3_df = l3.filter_batch(l2_df)
    passed = l3_df["passed"].sum()
    log(f"  L3 完成: {len(l3_df)} 候选, {passed} 通过")

    qp = None
    if QlibPredictor is not None:
        try:
            qp = QlibPredictor()
        except Exception:
            pass
    l4 = L4Ranker(qlib_predictor=qp)
    l4_df = l4.rank(l3_df)
    log(f"  L4 完成: {len(l4_df)} 只排名 (最高 rank={l4_df['global_rank'].max() if not l4_df.empty else 0})")
    return l4_df


# ── 阶段 5: 回测 ──────────────────────────────────────────────────

def backtest_top_n(l4_df: pd.DataFrame, signal_date: str, top_n: int = 100) -> pd.DataFrame:
    if l4_df.empty:
        log("  回测: L4 结果为空, 跳过")
        return pd.DataFrame()

    top = l4_df.head(top_n)
    trades = []

    for i, (_, row) in enumerate(top.iterrows()):
        code = row["code"]
        buy_type = row["buy_type"]
        try:
            daily = load_daily(code)
            if daily is None:
                continue
            sig_df = load_signals(code)
            if sig_df is None:
                continue
            struct_df = load_structure_for_code(code)

            changes = detect_all_changes(sig_df)
            buy_events = [c for c in changes if c["type"] == "buy"]
            sell_events = [c for c in changes if c["type"] == "sell"]

            target_d = pd.Timestamp(signal_date).date()
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

            engine = ExitEngine(code, entry_price, entry_date, matched["signal_label"], struct_df)

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
                    break

            if exit_price_val is None:
                last_bar = window.iloc[-1]
                exit_date = last_bar["date"]
                exit_price_val = float(last_bar["open"])
                exit_reason = "到期"
                weighted_ret = engine.compute_weighted_return(entry_price, exit_price_val)

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
                "l4_rank": int(row.get("global_rank", i + 1)),
            })
        except Exception:
            pass

        if (i + 1) % 20 == 0:
            log(f"  回测 {i+1}/{min(top_n, len(l4_df))}, 已成交 {len(trades)}")

    return pd.DataFrame(trades)


# ── 汇总 ──────────────────────────────────────────────────────────

def report(results: dict):
    print("\n" + "=" * 70)
    print("  烟雾测试报告")
    print("=" * 70)

    for dt, data in results.items():
        df = data["trades"]
        candidates = data["candidates"]
        l2_pass = data["l2_pass"]
        l3_pass = data.get("l3_pass", "?")
        l4_total = data["l4_total"]

        print(f"\n📅 {dt}")
        print(f"  候选 (当天买点): {candidates} 只")
        print(f"  L2 通过: {l2_pass} 只")
        print(f"  L3 通过: {l3_pass} 只")
        print(f"  L4 排名: {l4_total} 只")

        if df.empty:
            print("  ⚠️ 无成交记录")
            continue

        n = len(df)
        win = (df["return_pct"] > 0).sum()
        avg_ret = df["return_pct"].mean()
        med_ret = df["return_pct"].median()
        total_ret = df["return_pct"].sum()

        print(f"  top{n} 回测:")
        print(f"    成交: {n} 笔, 胜率: {win}/{n} ({win/n*100:.1f}%)")
        print(f"    平均收益: {avg_ret:+.2f}%, 中位数: {med_ret:+.2f}%")
        print(f"    累计收益: {total_ret:+.2f}%")

        reasons = df["exit_reason"].value_counts()
        print("    出场原因:")
        for r, c in reasons.items():
            print(f"      {r}: {c} 笔 ({c/n*100:.1f}%)")

        print(f"\n  前 5 笔:")
        for _, t in df.head(5).iterrows():
            print(f"    {t['code']} {t['buy_type']} 入场{t['entry_date']} "
                  f"→ {t['exit_date']} {t['return_pct']:+.2f}% ({t['exit_reason']})")

    print("\n" + "=" * 70)


# ── 主入口 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="烟雾测试 — 信号节点 → L1-L4 → top100 回测")
    parser.add_argument("dates", nargs="+", help="信号日期 YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=100, help="回测 Top N (默认 100)")
    parser.add_argument("--regime", default="CHOP", choices=["CHOP", "BEAR", "BULL"],
                       help="Regime 模式 (默认 CHOP)")
    args = parser.parse_args()

    all_results = {}
    total_start = time.time()

    for signal_date in args.dates:
        log(f"\n{'='*50}\n📅 信号节点: {signal_date}\n{'='*50}")
        t0 = time.time()

        log("阶段 1/5: 快速预扫当天买点信号…")
        candidates = fast_scan_buys(signal_date)
        if not candidates:
            log(f"⚠️ {signal_date}: 无买点信号")
            continue

        auto_regime = detect_regime(signal_date)
        log(f"  自动检测 mode: {auto_regime}")
        log(f"阶段 2/5: L2 EntryFilter ({len(candidates)} 候选) …")
        l2_df = run_l2(candidates, regime=auto_regime)

        log("阶段 3-4/5: L3 过滤 + L4 排名 …")
        l4_df = run_l34(l2_df, regime=auto_regime)

        actual_n = min(args.top, len(l4_df))
        log(f"阶段 5/5: 回测 top {actual_n} …")
        trades_df = backtest_top_n(l4_df, signal_date, top_n=actual_n)

        elapsed = time.time() - t0
        log(f"✅ {signal_date} 完成 ({elapsed:.0f}s)")

        # L3 pass count from l4_df metadata
        l3_passed = l4_df.attrs.get("total_l3", "?") if not l4_df.empty else 0

        all_results[signal_date] = {
            "candidates": len(candidates),
            "l2_pass": int(l2_df["passed"].sum()) if not l2_df.empty else 0,
            "l3_pass": l3_passed,
            "l4_total": len(l4_df),
            "trades": trades_df,
        }

    total_elapsed = time.time() - total_start
    log(f"\n全部完成, 总耗时 {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    report(all_results)

    out = BASE_DIR / "tmp_out" / "smoke_test_results.csv"
    all_trades = []
    for dt, d in all_results.items():
        if not d["trades"].empty:
            df_copy = d["trades"].copy()
            df_copy["信号节点"] = dt
            all_trades.append(df_copy)
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(out, index=False, encoding="utf-8-sig")
        log(f"\n详细数据已保存: {out}")


if __name__ == "__main__":
    main()
