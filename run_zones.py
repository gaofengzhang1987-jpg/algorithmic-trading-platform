#!/usr/bin/env python3
"""一键执行四级漏斗全流程 L1 -> L2 -> L3 -> L4（含 B+ 直通分叉）。"""

import sys
import time
import logging

import pandas as pd
from pathlib import Path

from zone1_deposition import run as l1_run
from verify_buy_type import verify_buy_type
from zone2_regime import run as l2_run
from zone3_regime import run as l3_run
from zone4_regime import run as l4_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_zones")




def main():
    t0 = time.time()

    # Auto regime detection（基于上证指数 MA 状态）
    def _detect_regime():
        try:
            idx = pd.read_parquet(Path(__file__).parent / "data" / "index" / "000001.parquet")
            idx = idx.sort_values("date")
            if len(idx) < 60: return "CHOP"
            c = idx["close"].iloc[-1]
            m20 = idx["close"].rolling(20).mean().iloc[-1]
            m60 = idx["close"].rolling(60).mean().iloc[-1]
            if len(idx) >= 26:
                m20_prev = idx["close"].rolling(20).mean().iloc[-6]
                s = (m20 - m20_prev) / m20_prev * 100 if m20_prev > 0 else 0
            else: s = 0
            if c > m20 > m60 and s > 0: return "BULL"
            if c < m20 < m60 and s < 0: return "BEAR"
            return "CHOP"
        except Exception:
            return "CHOP"

    regime = sys.argv[2] if len(sys.argv) > 2 else _detect_regime()
    logger.info("Auto regime: %s (pass sys.argv[2] to override)", regime)
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    # L1: 全量沉淀区（已含拆行 + 标签）
    logger.info("=" * 40)
    logger.info("L1 全量沉淀区: 扫描全市场买点信号...")
    df1 = l1_run()
    if df1.empty:
        logger.info("L1 无候选，流程终止")
        return
    logger.info("L1 完成: %d 只 (%.1fs)", len(df1), time.time() - t0)

    # L2: 计算 B+ 通过股票，不设阈值
    bplus_codes = set()
    for _, row in df1.iterrows():
        code = row["代码"]
        label = row["买点类型"]
        bt = "一买" if "一买" in label else ("二买" if "二买" in label else "三买")
        if bt != "一买" and verify_buy_type(code, bt):
            bplus_codes.add((code, bt))
    logger.info("B+ 通过: %d 只 (二买+三买) 不设阈值", len(bplus_codes))

    # L2: Regime 路由
    logger.info("=" * 40)
    logger.info("L2 Regime Filter: regime=%s, candidates=%d", regime, len(df1))
    df2 = l2_run(df1, regime=regime, bplus_codes=bplus_codes)
    logger.info("L2 完成: %d 只 (%.1fs)", len(df2), time.time() - t0)

    # L3: 策略二区
    logger.info("=" * 40)
    logger.info("L3 策略二区: 应用策略二区规则...")
    df3 = l3_run(df2, regime=regime)
    logger.info("L3 完成: %d 只 (%.1fs)", len(df3), time.time() - t0)

    # L4: 购买建议区
    logger.info("=" * 40)
    logger.info("L4 购买建议区: 评分排名...")
    df4 = l4_run(df3, top_n=top_n)
    logger.info("L4 完成: %d 只 (%.1fs)", len(df4), time.time() - t0)

    # 漏斗摘要
    logger.info("=" * 40)
    logger.info("漏斗摘要: %d -> %d -> %d -> %d (%.0fs)",
                len(df1), len(df2), len(df3), len(df4), time.time() - t0)
    logger.info("Regime: %s", regime)

    if not df4.empty:
        print("\n=== L4 TOP 10 ===")
        cols = [c for c in ["代码", "买点类型", "L2_综合得分", "L2_Regime"] if c in df4.columns]
        print(df4[cols].head(10).to_string())


if __name__ == "__main__":
    main()
