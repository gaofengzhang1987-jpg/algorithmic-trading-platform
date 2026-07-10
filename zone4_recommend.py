#!/usr/bin/env python3
"""L4 购买建议区 — 权重优化的 7 维加权评分。

从 L3 输出中读取得分列，按 holdout 验证的最优权重加权排序。
最优权重来自 Phase 6 混合优化 + 样本外验证 (TOP 100):
  w1(笔数)=7, w2(MACD)=12, w3(背驰)=10, w4(底分型)=13,
  w5(共振)=19, w6(周线)=22, w7(30min)=17

输出: data/zones/L4_recommend.parquet
"""

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zone4")

BASE = Path(__file__).parent; ZONES = BASE / "data" / "zones"

# Phase 6 最优权重 (holdout verified)
L4_WEIGHTS = {"L2_笔数得分": 7, "L2_MACD得分": 12, "L2_背驰得分": 10, "L2_底分型得分": 13, "L3_共振类型": 19, "L3_周线止跌": 22, "L3_30min二买": 17}

# 归一化 (确保 sum=100)
_total = sum(L4_WEIGHTS.values())
for k in L4_WEIGHTS:
    L4_WEIGHTS[k] = L4_WEIGHTS[k] / _total

def run(input_df=None, top_n=100):
    if input_df is None:
        p = ZONES / "L3_watchlist.parquet"
        if not p.exists():
            logger.warning("L3 不存在")
            return pd.DataFrame()
        input_df = pd.read_parquet(p)

    if input_df.empty: return pd.DataFrame()

    df = input_df.copy()

    # 加权综合得分
    df["L4_综合评分"] = 0.0
    for col, weight in L4_WEIGHTS.items():
        if col in df.columns:
            df["L4_综合评分"] += df[col].fillna(0) * weight
        else:
            logger.debug("缺少列: %s", col)

    # 按综合评分降序排列
    df.sort_values("L4_综合评分", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) > top_n:
        df = df.head(top_n)

    out = ZONES / "L4_recommend.parquet"
    df.to_parquet(out)
    logger.info("L4 购买建议区: %d 只 → %s (权重: %s)", len(df), out,
                {k: round(v*100) for k, v in L4_WEIGHTS.items()})
    return df

if __name__ == "__main__":
    df = run()
    if df.empty: print("L4: 无结果")
    else:
        print(f"L4: {len(df)} 只")
        cols = ["代码", "现价", "买点类型", "L4_综合评分", "L3_共振类型"]
        print(df[cols].head(15).to_string())
