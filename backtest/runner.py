#!/usr/bin/env python3
"""缠论信号回测引擎 — 批量回测入口。"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.constants import (
    COMMISSION, CAPITAL_PER_TRADE, MAX_HOLD_DAYS, STOP_LOSS_PCT,
    HALF_CUT_TIMEOUT, SIGNALS_DIR,
)
from core.data import load_daily, load_signals, get_next_trading_day, get_price_at_date
from core.signal_detector import detect_all_changes
from core.structure_cache import load_structure_cache, load_structure_for_code
from core.metrics import compute_metrics
from backtest.exit_engine import ExitEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")


def simulate_trades(
    code: str,
    daily: pd.DataFrame,
    changes: list[dict],
    struct_df: pd.DataFrame | None = None,
    max_trades: int = 15,
) -> list[dict]:
    """单只股票回测 — 调用方负责传数据。

    Exit priority (per bar, earliest wins):
      1. Structure stop: close <= defense -> full exit
      2. V-drop: close < last up-pivot GG -> full exit
      3. FULL -> half-cut (top-fx + divergence + new high)
      4. HALF -> buyback / second-sell confirm / 30-day timeout
      5. Sell signal: next trading day open
    """
    buy_events = [c for c in changes if c["type"] == "buy"]
    sell_events = [c for c in changes if c["type"] == "sell"]
    if not buy_events:
        return []

    daily_sorted = daily.sort_values("date").reset_index(drop=True)
    trades = []

    for buy in buy_events:
        entry_date = get_next_trading_day(buy["date"], daily_sorted)
        if entry_date is None:
            continue
        entry_price = get_price_at_date(entry_date, daily_sorted)
        if entry_price is None or entry_price <= 0:
            continue

        engine = ExitEngine(code, entry_price, entry_date, buy["signal_label"], struct_df)

        sell_exit_target = None
        sell_exit_label = ""
        for sell in sell_events:
            sell_date = pd.Timestamp(sell["date"])
            if sell_date > entry_date:
                cand = get_next_trading_day(sell["date"], daily_sorted)
                if cand is not None:
                    sell_exit_target = cand
                    sell_exit_label = sell["signal_label"]
                break

        state = "FULL"
        half_cut_fx_high = 0
        half_cut_day_count = 0
        position_pct = 1.0

        exit_date = None
        exit_price = None
        exit_reason = ""

        entry_idx = daily_sorted[daily_sorted["date"] == entry_date].index
        if entry_idx.empty:
            continue
        entry_idx = entry_idx[0]

        end_idx = min(entry_idx + MAX_HOLD_DAYS + 1, len(daily_sorted))
        window = daily_sorted.iloc[entry_idx + 1 : end_idx]
        if len(window) == 0:
            continue

        earliest_exit = None

        for bi in range(len(window)):
            bar = window.iloc[bi]
            bar_date = bar["date"]
            bar_close = bar["close"]
            bar_high = bar["high"]

            fx_on_bar = None
            if struct_df is not None and not struct_df.empty:
                bar_d = bar_date.date() if hasattr(bar_date, 'date') else bar_date
                if isinstance(bar_d, pd.Timestamp):
                    bar_d = bar_d.date()
                fx_rows = struct_df[
                    (pd.to_datetime(struct_df["edt"]).dt.date == bar_d) &
                    (struct_df["fx_b_mark"].str.contains("\u5206\u578b", na=False))
                ]
                if len(fx_rows) > 0:
                    frow = fx_rows.iloc[-1]
                    fx_on_bar = {
                        "mark": str(frow["fx_b_mark"]),
                        "high": float(frow.get("fx_b_high", frow.get("high", 0))),
                        "low": float(frow.get("fx_b_low", frow.get("low", 0))),
                    }

            if state != "EMPTY":
                engine.update_defense(bar_date)

            if state != "EMPTY" and engine.defense > 0 and bar_close <= engine.defense:
                earliest_exit = (bi, "\u7ed3\u6784\u6b62\u635f", bar_close, position_pct)
                break

            if state != "EMPTY" and engine.check_v_drop(bar_date, bar_close):
                earliest_exit = (bi, "V\u578b\u66b4\u8dcc\u7a7fGG", bar_close, position_pct)
                break

            if state == "FULL":
                half_cut = engine.check_half_cut(bar_date, bar_high, fx_on_bar)
                if half_cut:
                    half_cut_fx_high = half_cut["fx_high"]
                    half_cut_day_count = 0
                    position_pct = 0.5
                    state = "HALF"

            elif state == "HALF":
                half_cut_day_count += 1

                if engine.check_buyback(bar_date, bar_high, half_cut_fx_high):
                    position_pct = 1.0
                    state = "FULL"
                    engine.rebind_defense(bar_date)
                    half_cut_day_count = 0
                    continue

                if half_cut_day_count > 1:
                    start_i = max(0, bi - half_cut_day_count + 1)
                    max_high_since = window["high"].iloc[start_i : bi + 1].max()
                    if max_high_since < half_cut_fx_high:
                        second_sell = engine.check_second_sell(bar_date, bar_high, half_cut_fx_high, fx_on_bar)
                        if second_sell:
                            earliest_exit = (bi, "\u4e8c\u5356\u786e\u8ba4", bar_close, position_pct)
                            break

                if half_cut_day_count >= HALF_CUT_TIMEOUT:
                    earliest_exit = (bi, "\u534a\u4ed3\u8d85\u65f6", bar_close, position_pct)
                    break

            if sell_exit_target is not None and bar_date >= sell_exit_target:
                sp = get_price_at_date(bar_date, daily_sorted) or bar_close
                earliest_exit = (bi, "\u5356\u70b9", sp, position_pct)
                break

        if earliest_exit is not None:
            exit_idx, exit_reason, exit_price_val, _ = earliest_exit
            exit_date = window.iloc[exit_idx]["date"]
            exit_price = exit_price_val
            exit_reason = exit_reason
        else:
            exit_date = window.iloc[-1]["date"]
            exit_price = float(window.iloc[-1]["open"])
            exit_reason = "\u5230\u671f"

        if exit_price is None and exit_date is not None:
            exit_price = get_price_at_date(exit_date, daily_sorted)
        if exit_price is None or exit_price <= 0:
            continue

        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - 2 * COMMISSION
        hold_days = (exit_date - entry_date).days

        trades.append({
            "code": code,
            "signal_type": buy["signal_label"],
            "signal_date": buy["date"],
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "return_pct": round(net_return * 100, 2),
            "hold_days": hold_days,
            "exit_reason": exit_reason,
        })

    return trades


def run_all(codes: list[str] | None = None) -> pd.DataFrame:
    if codes is None:
        codes = sorted(p.stem for p in SIGNALS_DIR.glob("*.parquet"))

    struct_cache = load_structure_cache()  # 全局汇总表
    all_trades = []
    for i, code in enumerate(codes):
        try:
            daily = load_daily(code)
            if daily is None:
                continue
            sig_df = load_signals(code)
            if sig_df is None:
                continue
            changes = detect_all_changes(sig_df)
            trades = simulate_trades(code, daily, changes, load_structure_for_code(code))
            all_trades.extend(trades)
        except Exception as e:
            logger.warning("%s: 回测失败: %s", code, str(e)[:80])
        if (i + 1) % 200 == 0:
            logger.info("回测进度: %d/%d, 累计交易 %d 笔", i + 1, len(codes), len(all_trades))

    df = pd.DataFrame(all_trades)
    if df.empty:
        logger.info("回测结果为空")
        return df

    df.sort_values("entry_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("回测完成: %d 只股票, %d 笔交易", len(set(df["code"])), len(df))
    return df


def main():
    import sys
    codes_arg = None
    if len(sys.argv) > 1:
        codes_arg = sys.argv[1:]
    df = run_all(codes_arg)
    if df.empty:
        print("无交易记录")
        return
    metrics = compute_metrics(df.to_dict("records"))
    print("=== 回测绩效 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n前 10 笔交易:")
    print(df.head(10)[["code", "signal_type", "entry_date", "exit_date", "return_pct", "exit_reason"]].to_string())


if __name__ == "__main__":
    main()
