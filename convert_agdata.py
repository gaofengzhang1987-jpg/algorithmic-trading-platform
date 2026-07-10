#!/usr/bin/env python3
"""AGdata 15 分钟数据 → 项目标准格式。

输出: data/min15/{code}.parquet, data/min30/{code}.parquet (2020+)
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_agdata")

SRC_DIR = Path.home() / "Desktop" / "stock_15min"
DST_DIR = Path(__file__).parent / "data"
MIN15_DIR = DST_DIR / "min15"
MIN30_DIR = DST_DIR / "min30"
SDT = "2020-01-01"


def convert_one(src: Path) -> tuple[int, int]:
    code_raw = src.stem.split(".")[0]
    try:
        df = pd.read_parquet(src)
    except Exception as e:
        logger.warning("%s: 读取失败 %s", code_raw, e)
        return 0, 0

    df = df.loc[df.index.get_level_values("trade_date") >= SDT]

    if df.empty:
        return 0, 0

    dr = df.reset_index()
    dr["code"] = code_raw
    dr.rename(columns={"trade_time": "date", "vol": "volume"}, inplace=True)
    min15 = dr[["date", "open", "high", "low", "close", "volume", "amount", "code"]].copy()
    min15.sort_values("date", inplace=True)
    min15.reset_index(drop=True, inplace=True)

    # 30 分钟合成
    tmp = min15.set_index("date")
    min30 = tmp.resample("30min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "amount": "sum",
        "code": "first",
    }).dropna(subset=["open"]).reset_index()

    n15, n30 = len(min15), len(min30)
    if n15 > 0:
        min15.to_parquet(MIN15_DIR / f"{code_raw}.parquet", index=False)
    if n30 > 0:
        min30.to_parquet(MIN30_DIR / f"{code_raw}.parquet", index=False)
    return n15, n30


def main():
    MIN15_DIR.mkdir(parents=True, exist_ok=True)
    MIN30_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC_DIR.glob("*.parquet"))
    logger.info("找到 %d 个文件", len(files))

    total15 = total30 = 0
    for i, f in enumerate(files):
        n15, n30 = convert_one(f)
        total15 += n15
        total30 += n30
        if (i + 1) % 20 == 0:
            logger.info("%d/%d", i + 1, len(files))

    logger.info("完成: %d 只, 15min=%d 30min=%d", len(files), total15, total30)

    # 验证输出
    for label, d in [("15分钟", MIN15_DIR), ("30分钟", MIN30_DIR)]:
        fs = list(d.glob("*.parquet"))
        if fs:
            df = pd.read_parquet(fs[0])
            logger.info("%s 示例 %s: %d 行, %s ~ %s",
                        label, fs[0].stem, len(df),
                        str(df["date"].iloc[0])[:16],
                        str(df["date"].iloc[-1])[:16])


if __name__ == "__main__":
    main()
