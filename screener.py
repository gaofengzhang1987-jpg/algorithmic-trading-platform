#!/usr/bin/env python3
"""czsc 信号筛选器 — 基于 czsc 信号 Parquet 的状态变化筛选。

czsc 信号是状态式的：每个信号列每根 K 线都有一个字符串值。
筛选逻辑：查找最近 N 根 K 线中状态变化的信号（即"信号触发"）。
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("screener")

SIGNALS_DIR = Path(__file__).parent / "data" / "signals"

# ——— 关注的信号列（直接从 czsc 信号 DataFrame 的列名匹配） ———
# 买点信号列 key（包含这些关键词的列）
BUY_SIGNAL_KEYS = ["BUY1", "BUY2", "看多", "金叉", "多头", "强势", "低量柱"]
# 卖点信号列 key
SELL_SIGNAL_KEYS = ["SELL1", "SELL2", "看空", "死叉", "空头", "弱势"]


def is_buy_column(col: str) -> bool:
    """判断信号列是否为买点相关。"""
    return any(k in col for k in BUY_SIGNAL_KEYS)


def is_sell_column(col: str) -> bool:
    """判断信号列是否为卖点相关。"""
    return any(k in col for k in SELL_SIGNAL_KEYS)


def normalize_state(state: str) -> str:
    """去掉信号值中的 _0 和 _任意 后缀，提取核心状态。"""
    if pd.isna(state):
        return "无"
    parts = state.rsplit("_", 2)  # 去掉 _v2_v3 尾缀
    return parts[0] if len(parts) >= 2 else state


def load_last_signal_state(code: str) -> dict | None:
    """加载单只股票的最新信号状态。

    Returns:
        {signal_column: current_state, ...} 或 None
    """
    p = SIGNALS_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None

    # 取最后一行（最新状态）
    last_row = df.iloc[-1]

    # 筛选信号列（排除基础行情列）
    base_cols = {"symbol", "id", "dt", "open", "close", "high", "low", "vol", "amount", "year"}
    signal_cols = [c for c in df.columns if c not in base_cols]

    states = {}
    for col in signal_cols:
        val = last_row[col]
        if pd.notna(val) and str(val) != "0":
            states[col] = str(val)

    return states


def detect_recent_triggers(code: str, lookback: int = 5) -> list[dict]:
    """检测最近 N 根 K 线内的状态变化（信号触发）。

    Returns:
        [{column, old_state, new_state, date, type: "buy"|"sell"}, ...]
    """
    p = SIGNALS_DIR / f"{code}.parquet"
    if not p.exists():
        return []
    try:
        df = pd.read_parquet(p)
    except Exception:
        return []

    base_cols = {"symbol", "id", "dt", "open", "close", "high", "low", "vol", "amount", "year"}
    signal_cols = [c for c in df.columns if c not in base_cols]

    if len(df) < lookback + 1:
        return []

    triggers = []
    recent = df.tail(lookback + 1)  # 取最近 lookback+1 根 K 线

    for col in signal_cols:
        values = recent[col].astype(str).tolist()
        # 找状态变化：相邻两行值不同
        for i in range(1, len(values)):
            old_v = normalize_state(values[i - 1])
            new_v = normalize_state(values[i])
            if old_v != new_v and new_v != "无" and "0" not in new_v:
                sig_type = "buy" if is_buy_column(col) else "sell" if is_sell_column(col) else "unknown"
                triggers.append({
                    "column": col,
                    "old_state": old_v,
                    "new_state": new_v,
                    "date": str(recent.iloc[i]["dt"].date()),
                    "type": sig_type,
                })

    return triggers


def screen(max_results: int = 200, signal_type: str = "all", lookback_days: int = 20) -> pd.DataFrame:
    """执行选股筛选。

    Args:
        max_results: 最大结果数
        signal_type: "buy" | "sell" | "all"
        lookback_days: 回溯天数

    Returns:
        候选股票 DataFrame
    """
    signal_files = sorted(SIGNALS_DIR.glob("*.parquet"))
    logger.info("扫描 %d 个信号文件", len(signal_files))

    rows = []
    for p in signal_files:
        code = p.stem
        triggers = detect_recent_triggers(code, lookback=lookback_days)

        if not triggers:
            continue

        # 按类型筛选
        if signal_type == "buy":
            triggers = [t for t in triggers if t["type"] == "buy"]
        elif signal_type == "sell":
            triggers = [t for t in triggers if t["type"] == "sell"]

        if not triggers:
            continue

        # 只取买点或卖点类型
        buy_triggers = [t for t in triggers if t["type"] == "buy"]
        sell_triggers = [t for t in triggers if t["type"] == "sell"]

        buy_summary = " | ".join(f"{t['new_state']}({t['date']})" for t in buy_triggers[:3])
        sell_summary = " | ".join(f"{t['new_state']}({t['date']})" for t in sell_triggers[:3])

        score = len(buy_triggers) * 10 + len(sell_triggers) * 2

        # 获取最新价格
        try:
            df = pd.read_parquet(p)
            last_price = float(df["close"].iloc[-1])
        except Exception:
            last_price = 0

        rows.append({
            "代码": code,
            "现价": round(last_price, 2),
            "买点信号": buy_summary or "-",
            "卖点信号": sell_summary or "-",
            "信号数": len(triggers),
            "评分": score,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.sort_values("评分", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) > max_results:
        df = df.head(max_results)

    logger.info("筛选完成: %d 只候选", len(df))
    return df


def main():
    df = screen()
    if df.empty:
        print("无候选")
        return
    print(df.to_string(max_rows=30))


if __name__ == "__main__":
    main()
