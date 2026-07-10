#!/usr/bin/env python3
"""区域二 监控塔 — 卖点监控 + L3/L4 细颗粒度监控。

职责：
  1. 非 L1 股票 → 扫描一卖/二卖/三卖信号
  2. L3/L4 股票 → 细颗粒度监控框架（占位，后续定义规则）

输出:
  - data/zones/monitor_sell.parquet  (卖点监控结果)
  - data/zones/monitor_watchlist.parquet (L3/L4 细监清单)
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("monitor")

BASE_DIR = Path(__file__).parent
SIGNALS_DIR = BASE_DIR / "data" / "signals"
ZONES_DIR = BASE_DIR / "data" / "zones"

# 卖点信号列的 key 标记
SELL_KEYS = ["SELL1", "BS2辅助V230320_二卖", "BS3辅助V230318_三卖", "BS3辅助V230319_三卖"]
# 细颗粒度监控的信号 key（多级别 + 辅助指标）
FINE_GRAIN_KEYS = ["SELL1", "BS2", "BS3", "三卖", "死叉", "空头", "弱势"]


def _is_sell_column(col: str) -> bool:
    return any(k in col for k in SELL_KEYS)


def _is_currently_sell(last_row, signal_cols: list[str]) -> bool:
    for col in signal_cols:
        if not _is_sell_column(col):
            continue
        val = str(last_row[col]) if not pd.isna(last_row[col]) else ""
        if "一卖" in val or "二卖" in val or "三卖" in val:
            return True
    return False


# ================================================================
#  区域二-A: 非 L1 股票卖点监控
# ================================================================

def scan_sell_signals(l1_codes: set[str], lookback: int = 20) -> pd.DataFrame:
    """扫描不在 L1 中的股票，检测卖点信号。

    Args:
        l1_codes: L1 中已有的股票代码（这些已被买点筛选，不再卖点监控）
        lookback: 回溯天数
    """
    signal_files = sorted(SIGNALS_DIR.glob("*.parquet"))
    if not signal_files:
        logger.warning("data/signals/ 目录为空")
        return pd.DataFrame()

    base_cols = {"symbol", "id", "dt", "open", "close", "high", "low", "vol", "amount", "year"}

    rows = []
    for p in signal_files:
        code = p.stem
        if code in l1_codes:
            continue  # 已在 L1 中的不重复监控卖点

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

        # 当前卖点状态
        last_state_sell = _is_currently_sell(last_row, signal_cols)

        # 状态变化检测
        sell_changes = []
        if len(df) >= lookback + 1:
            recent = df.tail(lookback + 1)
            for col in signal_cols:
                if not _is_sell_column(col):
                    continue
                values = recent[col].astype(str).tolist()
                for i in range(1, len(values)):
                    old_v = values[i - 1]
                    new_v = values[i]
                    if old_v != new_v and "0" not in new_v and new_v != "nan":
                        if "一卖" in new_v or "二卖" in new_v or "三卖" in new_v:
                            sell_changes.append({
                                "col": col,
                                "from": old_v.split("_")[0],
                                "to": new_v.split("_")[0],
                                "date": str(recent.iloc[i]["dt"].date()),
                            })

        if not sell_changes and not last_state_sell:
            continue

        if last_state_sell and not sell_changes:
            sell_changes.append({
                "col": "当前持有",
                "from": "一卖/二卖/三卖",
                "to": "预警中",
                "date": last_date,
            })

        # 提取卖点类型
        sell_zones = set()
        for col in signal_cols:
            val = str(last_row[col]) if not pd.isna(last_row[col]) else ""
            if "一卖" in val:
                sell_zones.add("一卖")
            if "二卖" in val:
                sell_zones.add("二卖")
            if "三卖" in val:
                sell_zones.add("三卖")

        # 辅助指标确认
        macd_dead = any(
            "死叉" in str(last_row[c]) for c in signal_cols
            if not pd.isna(last_row[c])
        )

        rows.append({
            "代码": code,
            "现价": round(last_price, 2),
            "最新日期": last_date,
            "卖点类型": " | ".join(sorted(sell_zones)),
            "MACD死叉": macd_dead,
            "信号数": len(sell_changes),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.sort_values("信号数", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_path = ZONES_DIR / "monitor_sell.parquet"
    df.to_parquet(out_path)
    logger.info("卖点监控: %d 只 -> %s", len(df), out_path)
    return df


# ================================================================
#  区域二-B: L3/L4 细颗粒度监控（框架）
# ================================================================

class FineGrainMonitor:
    """L3/L4 细颗粒度监控框架。

    当前为框架模式，仅记录被监控的股票。
    后续开发方向：
      - 多时间级别联立（日线 + 60分钟 + 15分钟）
      - 实时推送（飞书/钉钉/企微通知）
      - 持仓状态追踪（未触发→触发闪烁→确认后常亮）
      - 止损/止盈动态跟踪
    """

    def __init__(self):
        self.name = "细颗粒度监控框架"

    def watch(self, stocks_df: pd.DataFrame) -> pd.DataFrame:
        """对给定的股票列表建立监控清单。"""
        if stocks_df.empty:
            return pd.DataFrame()

        result = stocks_df[["代码", "现价"]].copy()
        result["监控级别"] = "细颗粒度"
        result["状态"] = "监控中"
        result["上次检查"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        return result


def run(l1_path: str | None = None,
        l3_path: str | None = None,
        l4_path: str | None = None):
    """执行完整监控流程。

    Returns:
        (sell_df, fine_grain_df)
    """
    # 读取 L1 代码列表（用于排除卖点监控）
    l1_codes = set()
    l1_file = l1_path or (ZONES_DIR / "L1_deposition.parquet")
    if Path(l1_file).exists():
        l1_df = pd.read_parquet(l1_file)
        l1_codes = set(l1_df["代码"].tolist())

    # 卖点监控
    sell_df = scan_sell_signals(l1_codes)

    # 细颗粒度监控
    fine_grain_df = pd.DataFrame()
    l3_file = l3_path or (ZONES_DIR / "L3_watchlist.parquet")
    if Path(l3_file).exists():
        monitor = FineGrainMonitor()
        l3_df = pd.read_parquet(l3_file)
        fine_grain_df = monitor.watch(l3_df)

    if not fine_grain_df.empty:
        out_path = ZONES_DIR / "monitor_watchlist.parquet"
        fine_grain_df.to_parquet(out_path)
        logger.info("细颗粒度监控: %d 只 -> %s", len(fine_grain_df), out_path)

    # 摘要
    logger.info("=" * 40)
    logger.info("监控摘要: 卖点预警 %d 只 | 细监 %d 只",
                len(sell_df), len(fine_grain_df))

    # 检查重叠：L1 中的股是否也出现在卖点监控
    l1_buy_codes = l1_codes
    sell_codes = set(sell_df["代码"].tolist()) if not sell_df.empty else set()
    overlap = l1_buy_codes & sell_codes
    if overlap:
        logger.warning("L1(买点) ∩ 卖点监控 = %d 只: %s", len(overlap), overlap)

    return sell_df, fine_grain_df


if __name__ == "__main__":
    sell_df, fine_df = run()
    if not sell_df.empty:
        print(f"\n=== 卖点监控 TOP10 ({len(sell_df)} 只) ===")
        print(sell_df.head(10).to_string())
    if not fine_df.empty:
        print(f"\n=== L3/L4 细监清单 ({len(fine_df)} 只) ===")
        print(fine_df.head(10).to_string())
