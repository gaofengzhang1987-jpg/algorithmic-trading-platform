#!/usr/bin/env python3
"""L1 全量沉淀区 — 从全市场信号中筛选含买点（一买/二买/三买）的股票。

检测逻辑：
  1. 最近 N 天内状态变化（从非买点 → 买点）
  2. 最新状态已处于买点（即使未在窗口内变化）

输出: data/zones/L1_deposition.parquet（按 L1_优先级分 降序，上限 1500）
"""

import logging
from pathlib import Path

import pandas as pd
from verify_buy_type import get_buy_label

MAX_L1 = 1500  # L1 池容量上限
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zone1")

BASE_DIR = Path(__file__).parent
SIGNALS_DIR = BASE_DIR / "data" / "signals"
ZONES_DIR = BASE_DIR / "data" / "zones"

# 买点信号列的 key 标记
BUY_KEYS = ["BUY1", "BS2辅助", "BS3辅助", "三买"]


# ── L1 优先级分档 ────────────────────────────────────────────
# 从买点类型后缀推导结构质量（A/B/C 三档），配合信号密度加成排序截断。
# 一买无后缀，纯信号密度驱动。

_TIER_MAP = {
    # A 档：B+ 结构验证通过
    "标准": "A", "浅回踩": "A",
    # B 档：结构有瑕疵但主体成立
    "类": "B", "弱突破": "B",
    # C 档：结构不成立
    "无基础": "C", "未企稳": "C", "创新低": "C",
    "回踩进中枢": "C", "跌破中枢": "C", "远离入场区": "C",
    "无中枢基础": "C",
}
_TIER_BASE = {"A": 100, "B": 60, "C": 20}


def _classify_tier(label: str) -> str:
    """从细分标签后缀映射到 A/B/C 档。

    一买无 B+ 验证后缀，固定归 B 档——结构有确认（CZSC 底分型成立）但未做完整验证。
    """
    if "一买" in label and "二买" not in label and "三买" not in label:
        return "B"
    for suffix, tier in _TIER_MAP.items():
        if suffix in label:
            return tier
    return "C"  # 未匹配的默认 C 档


def _compute_l1_priority(label: str, signal_count: int) -> int:
    """L1 优先级分数：三档基础分 + 信号密度加成，用于排序截断。

    一买固定 B 档（60）+ min(signal_count,10)×5，上限 110。
    二买/三买按后缀入 A/B/C 档 + min(signal_count,10)×5。
    """
    base = _TIER_BASE.get(_classify_tier(label), 20)
    density_bonus = min(signal_count, 10) * 5
    return base + density_bonus


def _categorize_priority(score: int) -> str:
    """优先级分数 → 可读分类标签，供排查参考。"""
    if score >= 100:
        return "A-高优先级"
    elif score >= 70:
        return "B-中等优先级"
    return "C-低优先级"


# ── 信号扫描 ─────────────────────────────────────────────────

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
        if not val or str(val) == "0":
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
                    if old_v != new_v and new_v != "0" and new_v != "nan":
                        if "一买" in new_v or "二买" in new_v or "三买" in new_v:
                            buy_changes.append({
                                "col": col,
                                "from": old_v.split("_")[0],
                                "to": new_v.split("_")[0],
                                "date": str(recent.iloc[i]["dt"].date()),
                                "from_is_buy": any(k in old_v for k in ("一买", "二买", "三买")),
                            })

        if not buy_changes:  # 仅通过 20 日内状态变化进入 L1
            continue

        latest_states = {c: str(df.iloc[-1][c]) if not pd.isna(df.iloc[-1][c]) else ""
                         for c in signal_cols}
        zones = _extract_zones(latest_states, signal_cols)
        if not zones:
            continue

        state_summary = {}
        for ch in buy_changes:
            key = ch["col"].split("_")[2] if "_" in ch["col"] else ch["col"][:20]
            state_summary[key] = f"{ch['to']}({ch['date']})"

        # 按买点类型拆行，每行打细分标签
        for bt in zones:
            try:
                label = get_buy_label(code, buy_type=bt)
            except Exception:
                label = bt
            rows.append({
                "代码": code,
                "现价": round(last_price, 2),
                "最新日期": last_date,
                "买点类型": label,
                "状态详情": " | ".join(f"{k}:{v}" for k, v in list(state_summary.items())[:5]),
                "信号数": len(buy_changes),
                "非买点转买点": any(
                    ch["to"] == bt and not ch["from_is_buy"] for ch in buy_changes
                ),
                "当天非买转买": any(
                    ch["to"] == bt
                    and ch["date"] == last_date
                    and not ch["from_is_buy"]
                    for ch in buy_changes
                ),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.info("L1: 无候选")
        return df

    # ── L1 优先级排序 + 截断 ─────────────────────────────────
    df["L1_优先级分"] = df.apply(
        lambda r: _compute_l1_priority(r["买点类型"], r["信号数"]), axis=1
    )
    df["L1_优先级"] = df["L1_优先级分"].apply(_categorize_priority)
    df.sort_values("L1_优先级分", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    original_count = len(df)
    if original_count > MAX_L1:
        cutoff_score = df.iloc[MAX_L1 - 1]["L1_优先级分"]
        df = df.head(MAX_L1).reset_index(drop=True)
        logger.info("L1 截断: %d → %d (上限 %d, 截断线分数=%d)",
                    original_count, MAX_L1, MAX_L1, cutoff_score)

    out_path = ZONES_DIR / "L1_deposition.parquet"
    df.to_parquet(out_path)
    logger.info("L1 全量沉淀区: %d 只 -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    df = run()
    if not df.empty:
        print(f"L1: {len(df)} 只")
        cols = [c for c in ["代码", "现价", "买点类型", "信号数", "L1_优先级分", "L1_优先级"]
                if c in df.columns]
        print(df[cols].head(20).to_string())
