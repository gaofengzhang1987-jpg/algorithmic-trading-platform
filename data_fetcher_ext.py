#!/usr/bin/env python3
"""扩展数据拉取 — 周线 + 30分钟 + MACD。

周线：从日线 pandas resample('W') 合成
30分钟：从 Sina 5分钟数据合成（akshare.stock_zh_a_minute）
MACD：czsc generate_czsc_signals 已包含 MACD 信号
换手率：Sina 日线已含 turnover 列
"""

import logging
import time
from pathlib import Path

import pandas as pd
import akshare as ak
from czsc import RawBar, Freq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_fetcher_ext")

BASE_DIR = Path(__file__).parent
DAILY_DIR = BASE_DIR / "data" / "daily"
WEEKLY_DIR = BASE_DIR / "data" / "weekly"
MIN30_DIR = BASE_DIR / "data" / "min30"

for d in [WEEKLY_DIR, MIN30_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ================================================================
#  周线合成
# ================================================================

def build_weekly(code: str) -> pd.DataFrame | None:
    """从日线重采样生成周线。"""
    p = DAILY_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    weekly = df.resample("W").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
        "code": "first",
    })
    weekly.dropna(inplace=True)
    weekly.reset_index(inplace=True)
    return weekly


def build_all_weekly(codes: list[str] | None = None):
    if codes is None:
        codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
    for code in codes:
        w = build_weekly(code)
        if w is not None:
            w.to_parquet(WEEKLY_DIR / f"{code}.parquet", index=False)
    logger.info("周线合成: %d 只", len(list(WEEKLY_DIR.glob("*.parquet"))))


# ================================================================
#  30分钟数据 (从 Sina 5分钟合成)
# ================================================================

def fetch_min30(code: str, days: int = 30) -> pd.DataFrame | None:
    """拉取最近 N 天的 5分钟数据，合成 30分钟。

    注: Sina stock_zh_a_minute 只提供近期分钟数据（约 5-10 个交易日）。
    """
    sina_code = f"sz{code}" if code.startswith(("0", "3")) else f"sh{code}"
    try:
        df = ak.stock_zh_a_minute(symbol=sina_code, period="5")
    except Exception as e:
        logger.debug("%s 拉取分钟失败: %s", code, str(e)[:60])
        return None

    if df is None or df.empty:
        return None

    # 合成 30 分钟 K 线
    df["day"] = pd.to_datetime(df["day"])
    df.set_index("day", inplace=True)

    min30 = df.resample("30min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
    })
    min30.dropna(inplace=True)
    min30["code"] = code
    min30.reset_index(inplace=True)
    min30.rename(columns={"day": "date"}, inplace=True)
    return min30


def fetch_all_min30(codes: list[str] | None = None):
    if codes is None:
        codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
    success = 0
    for i, code in enumerate(codes):
        m30 = fetch_min30(code)
        if m30 is not None and not m30.empty:
            m30.to_parquet(MIN30_DIR / f"{code}.parquet", index=False)
            success += 1
        if (i + 1) % 500 == 0:
            logger.info("30分钟进度: %d/%d, 成功 %d", i + 1, len(codes), success)
    logger.info("30分钟拉取: %d/%d", success, len(codes))


# ================================================================
#  MACD 列 (来自 czsc 信号引擎，无需额外拉取)
# ================================================================

def macd_status():
    """MACD 数据已通过 generate_czsc_signals 计算，列名格式：
    日线_D1MACD12#26#9_BS1辅助V230313
    """
    signal_files = list((BASE_DIR / "data" / "signals").glob("*.parquet"))
    if not signal_files:
        return "信号文件尚未生成，MACD 列会在 signal_engine.py 完成计算后出现"
    return f"已有 {len(signal_files)} 个信号文件，每个包含 MACD 金叉/死叉状态列"


# ================================================================
#  CLI
# ================================================================

def main():
    import sys

    if "--weekly" in sys.argv:
        logger.info("合成周线...")
        build_all_weekly()
    elif "--min30" in sys.argv:
        logger.info("拉取30分钟数据...")
        fetch_all_min30()
    elif "--all" in sys.argv:
        logger.info("全量扩展数据...")
        build_all_weekly()
        fetch_all_min30()
    else:
        print("Usage:")
        print("  python3 data_fetcher_ext.py --weekly   # 合成周线")
        print("  python3 data_fetcher_ext.py --min30    # 拉取30分钟")
        print("  python3 data_fetcher_ext.py --all      # 全部")

    logger.info("MACD状态: %s", macd_status())
    logger.info("换手率: Sina 日线已含 turnover 列 ✓")


if __name__ == "__main__":
    main()
