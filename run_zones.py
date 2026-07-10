#!/usr/bin/env python3
"""一键执行四级漏斗全流程 L1 → L2 → L3 → L4。"""

import sys
import time
import logging

from zone1_deposition import run as l1_run
from zone2_pattern import run as l2_run
from zone3_watchlist import run as l3_run
from zone4_recommend import run as l4_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_zones")


def main():
    t0 = time.time()

    # L1: 全量沉淀区
    logger.info("=" * 40)
    logger.info("L1 全量沉淀区: 扫描全市场买点信号...")
    df1 = l1_run()
    if df1.empty:
        logger.info("L1 无候选，流程终止")
        return
    logger.info("L1 完成: %d 只 (%.1fs)", len(df1), time.time() - t0)

    # L2: 策略一区
    logger.info("=" * 40)
    logger.info("L2 策略一区: 应用策略一区规则...")
    df2 = l2_run(df1)
    logger.info("L2 完成: %d 只 (%.1fs)", len(df2), time.time() - t0)

    # L3: 策略二区
    logger.info("=" * 40)
    logger.info("L3 策略二区: 应用策略二区规则...")
    df3 = l3_run(df2)
    logger.info("L3 完成: %d 只 (%.1fs)", len(df3), time.time() - t0)

    # L4: 购买建议区
    logger.info("=" * 40)
    logger.info("L4 购买建议区: 评分排名...")
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    df4 = l4_run(df3, top_n=top_n)
    logger.info("L4 完成: %d 只 (%.1fs)", len(df4), time.time() - t0)

    # 漏斗摘要
    logger.info("=" * 40)
    logger.info("漏斗摘要: %d → %d → %d → %d (%.0fs)",
                len(df1), len(df2), len(df3), len(df4), time.time() - t0)

    if not df4.empty:
        print("\n=== L4 购买建议 TOP10 ===")
        print(df4.head(10).to_string())


if __name__ == "__main__":
    main()
