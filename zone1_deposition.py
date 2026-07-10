#!/usr/bin/env python3
"""L1 全量沉淀区 — 从全市场信号中筛选含买点（一买/二买/三买）的股票。

检测逻辑：
  1. 最近 N 天内状态变化（从非买点 → 买点）
  2. 最新状态已处于买点（即使未在窗口内变化）

输出: data/zones/L1_deposition.parquet
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zone1")

BASE_DIR = Path(__file__).parent
SIGNALS_DIR = BASE_DIR / "data" / "signals"
ZONES_DIR = BASE_DIR / "data" / "zones"

# 买点信号列的 key 标记
BUY_KEYS = ["BUY1", "BS2辅助", "BS3辅助", "三买"]


def _is_buy_column(col: str) -> bool:
    return any(k in col for k in BUY_KEYS)


def _is_currently_buy(last_row, signal_cols: list[str]) -> bool:
    for col in signal_cols:
        if not _is_buy_column(col):
            continue
        val = str(last_row[col]) if not pd.isna(last_row[col]) else ""
        if "一买" in val or "二买" in val or "三买" in val:
            return True
    return False


def _extract_zones(states: dict, signal_cols: list[str]) -> list[str]:
    zones = set()
    buy_cols = [c for c in signal_cols if _is_buy_column(c)]
    for col in buy_cols:
        val = states.get(col, "")
        if not val or "0" in str(val):
            continue
        if "一买" in val:
            zones.add("一买")
        if "二买" in val:
            zones.add("二买")
        if "三买" in val:
            zones.add("三买")
    return sorted(zones)


def run(lookback: int = 20) -> pd.DataFrame:
    signal_files = sorted(SIGNALS_DIR.glob("*.parquet"))
    if not signal_files:
        logger.warning("data/signals/ 目录为空")
        return pd.DataFrame()

    logger.info("扫描 %d 个信号文件", len(signal_files))
    base_cols = {"symbol", "id", "dt", "open", "close", "high", "low", "vol", "amount", "year"}

    rows = []
    for p in signal_files:
        code = p.stem
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue

        signal_cols = [c for c in df.columns if c not in base_cols]
        if len(df) < 2:
            continue

        last_row = df.iloc[-1]
        last_price = float(last_row["close"])
        last_date = str(last_row["dt"].date())

        # ① 当前状态检查
        last_state_buy = _is_currently_buy(last_row, signal_cols)

        # ② 状态变化检测
        buy_changes = []
        if len(df) >= lookback + 1:
            recent = df.tail(lookback + 1)
            for col in signal_cols:
                if not _is_buy_column(col):
                    continue
                values = recent[col].astype(str).tolist()
                for i in range(1, len(values)):
                    old_v = values[i - 1]
                    new_v = values[i]
                    if old_v != new_v and "0" not in new_v and new_v != "nan":
                        if "一买" in new_v or "二买" in new_v or "三买" in new_v:
                            buy_changes.append({
                                "col": col,
                                "from": old_v.split("_")[0],
                                "to": new_v.split("_")[0],
                                "date": str(recent.iloc[i]["dt"].date()),
                            })

        if not buy_changes and not last_state_buy:
            continue

        if last_state_buy and not buy_changes:
            buy_changes.append({
                "col": "当前持有",
                "from": "一买/二买/三买",
                "to": "持有中",
                "date": last_date,
            })

        latest_states = {c: str(df.iloc[-1][c]) if not pd.isna(df.iloc[-1][c]) else "" for c in signal_cols}
        zones = _extract_zones(latest_states, signal_cols)

        state_summary = {}
        for ch in buy_changes:
            key = ch["col"].split("_")[2] if "_" in ch["col"] else ch["col"][:20]
            state_summary[key] = f"{ch['to']}({ch['date']})"

        rows.append({
            "代码": code,
            "现价": round(last_price, 2),
            "最新日期": last_date,
            "买点类型": " | ".join(zones),
            "状态详情": " | ".join(f"{k}:{v}" for k, v in list(state_summary.items())[:5]),
            "信号数": len(buy_changes),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.info("L1: 无候选")
        return df

    df.sort_values("信号数", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_path = ZONES_DIR / "L1_deposition.parquet"
    df.to_parquet(out_path)
    logger.info("L1 全量沉淀区: %d 只 -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    df = run()
    if not df.empty:
        print(f"L1: {len(df)} 只")
        print(df[["代码", "现价", "买点类型", "信号数"]].head(20).to_string())
