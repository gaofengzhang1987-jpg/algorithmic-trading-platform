#!/usr/bin/env python3
"""L3 Regime Filter — wraps l3_filter.L3Filter for pipeline integration.

Replaces zone3_watchlist.py with regime-routed quality filtering.
Level 2 filters disabled across all regimes per user request.
"""

import logging
from pathlib import Path

import pandas as pd

from l3_filter import L3Filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zone3_regime")

BASE = Path(__file__).parent
ZONES = BASE / "data" / "zones"


def _parse_base_buy_type(label: str) -> str:
    """Extract simple buy type from label like '二买_标准' -> '二买'."""
    if "一买" in label: return "一买"
    if "二买" in label: return "二买"
    if "三买" in label: return "三买"
    return "一买"


def run(input_df=None, regime="CHOP"):
    if input_df is None:
        p = ZONES / "L2_regime.parquet"
        if not p.exists():
            logger.warning("L2 not found")
            return pd.DataFrame()
        input_df = pd.read_parquet(p)
    if input_df.empty:
        return pd.DataFrame()

    logger.info("L3 Regime Filter: regime=%s candidates=%d", regime, len(input_df))

    # Map L2 columns to l3_filter expected format
    l2_input = pd.DataFrame({
        "code": input_df["代码"],
        "buy_type": input_df["买点类型"].apply(_parse_base_buy_type),
        "signal_date": input_df["最新日期"],
        "total_score": input_df.get("L2_综合得分", 0),
    })

    # Run l3_filter
    flt = L3Filter(regime=regime)
    l3_result = flt.filter_batch(l2_input)

    # Count rejections
    passed_df = l3_result[l3_result["passed"]]
    rej_counts = l3_result[~l3_result["passed"]]["reject_reasons"].value_counts().to_dict()
    logger.info("L3 Regime: %d->%d (rejections: %s)",
                len(input_df), len(passed_df),
                {k: v for k, v in rej_counts.items()} if rej_counts else "none")

    if passed_df.empty:
        return pd.DataFrame()

    # Merge L3 filter results back with original L2 fields (exact code+buy_type match)
    passed_df["_key"] = passed_df["code"] + "|" + passed_df["buy_type"]
    input_df["_key"] = input_df["代码"] + "|" + input_df["买点类型"].apply(_parse_base_buy_type)
    result = input_df[input_df["_key"].isin(passed_df["_key"])].copy()
    result.drop(columns=["_key"], inplace=True)

    # Attach L3 filter dimensions
    for col in ["sector_rps", "stock_rps", "atr_pct", "vol_ratio", "high_dist"]:
        if col in passed_df.columns:
            lookup = passed_df.set_index("code")[col]
            result[col] = result["代码"].map(lookup)

    result.reset_index(drop=True, inplace=True)

    out_path = ZONES / "L3_regime.parquet"
    result.to_parquet(out_path)
    logger.info("L3 Regime: %d passed -> %s", len(result), out_path)
    return result


if __name__ == "__main__":
    df = run()
    if df.empty:
        print("L3 Regime: no passes")
    else:
        print(f"L3 Regime: {len(df)} passed")
