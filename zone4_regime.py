#!/usr/bin/env python3
"""L4 Regime Ranker — wraps l4_ranker.L4Ranker for pipeline integration."""

import logging
from pathlib import Path
import pandas as pd
from l4_ranker import L4Ranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zone4_regime")

BASE = Path(__file__).parent
ZONES = BASE / "data" / "zones"


def _parse_base_buy_type(label: str) -> str:
    if "一买" in label: return "一买"
    if "二买" in label: return "二买"
    if "三买" in label: return "三买"
    return "一买"


def run(input_df=None, top_n=100):
    if input_df is None:
        p = ZONES / "L3_regime.parquet"
        if not p.exists():
            logger.warning("L3 not found")
            return pd.DataFrame()
        input_df = pd.read_parquet(p)
    if input_df.empty:
        return pd.DataFrame()

    logger.info("L4 Ranker: candidates=%d", len(input_df))

    # Map to l4_ranker expected format
    l4_input = pd.DataFrame({
        "code": input_df["代码"],
        "buy_type": input_df["买点类型"].apply(_parse_base_buy_type),
        "passed": True,
        "total_score": input_df.get("L2_综合得分", 0).astype(float),
        "sector_rps": input_df.get("sector_rps", 0).astype(float),
        "stock_rps": input_df.get("stock_rps", 0).astype(float),
    })

    ranker = L4Ranker()
    ranked = ranker.rank(l4_input)

    if ranked.empty:
        logger.info("L4 Ranker: 0 passed after sector cull")
        return pd.DataFrame()

    # Attach price and original label from L3
    price_map = input_df.drop_duplicates(subset=["代码"]).set_index("代码")["现价"].to_dict()
    label_map = input_df.drop_duplicates(subset=["代码"]).set_index("代码")["买点类型"].to_dict()
    regime_map = input_df.drop_duplicates(subset=["代码"]).set_index("代码").get("L2_Regime", pd.Series("", index=input_df.index)).to_dict()

    ranked["现价"] = ranked["code"].map(price_map)
    ranked["买点类型"] = ranked["code"].map(label_map)
    ranked["L2_Regime"] = ranked["code"].map(regime_map)
    ranked["L2_综合得分"] = ranked["total_score"]

    # Dedup: keep max composite per code (already sorted by _zone + composite)
    ranked = ranked.drop_duplicates(subset=["code"], keep="first")
    ranked = ranked.head(top_n)

    logger.info("L4 Ranker: %d after dedup (★★★: %d)",
                len(ranked), (ranked.get("tag", "") == "★★★").sum() if "tag" in ranked.columns else 0)

    out_path = ZONES / "L4_recommend.parquet"
    ranked.to_parquet(out_path)
    return ranked


if __name__ == "__main__":
    df = run()
    if df.empty:
        print("L4: no results")
    else:
        print(f"L4: {len(df)} ranked")
        cols = [c for c in ["code", "买点类型", "composite", "zone_rank", "tag"] if c in df.columns]
        print(df[cols].head(15).to_string())
