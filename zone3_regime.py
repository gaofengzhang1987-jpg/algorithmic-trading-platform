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
STOCK_RPS = BASE / "data" / "reference" / "stock_rps.parquet"
INDUSTRY_RPS = BASE / "data" / "reference" / "industry_rps.parquet"
INDUSTRY_MAP = BASE / "data" / "industry_classification.parquet"


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

    # Triple-resonance (三级联立) bypass: skip L3, pass directly
    _triple_mask = input_df["买点类型"].str.contains("30m_结构联立", na=False)
    _triple_df = input_df[_triple_mask].copy()
    if len(_triple_df) > 0:
        logger.info("L3: %d triple-resonance stocks bypass L3", len(_triple_df))
        l2_input = l2_input[~_triple_mask.values]
    else:
        _triple_df = pd.DataFrame()

    # Run l3_filter
    flt = L3Filter(regime=regime)
    l3_result = flt.filter_batch(l2_input)

    # Count rejections
    passed_df = l3_result[l3_result["passed"]]
    rej_counts = l3_result[~l3_result["passed"]]["reject_reasons"].value_counts().to_dict()
    logger.info("L3 Regime: %d->%d (rejections: %s)",
                len(input_df), len(passed_df),
                {k: v for k, v in rej_counts.items()} if rej_counts else "none")

    # Keep triple-resonance stocks even if regular filtering killed everything
    if passed_df.empty and len(_triple_df) > 0:
        passed_df = pd.DataFrame({
            "code": _triple_df["代码"],
            "buy_type": _triple_df["买点类型"],
            "passed": True,
            "reject_reasons": "三级联立豁免",
            "total_score": _triple_df.get("L2_综合得分", 0),
        })
        # Attach RPS
        try:
            srps = pd.read_parquet(STOCK_RPS).sort_values("date").groupby("code").last().reset_index()
            irps = pd.read_parquet(INDUSTRY_RPS).sort_values("date").groupby("industry").last().reset_index()
            imap = pd.read_parquet(INDUSTRY_MAP).set_index("code")["industry"]
            _c = passed_df["code"].astype(str).str.zfill(6)
            _i = _c.map(imap)
            passed_df["sector_rps"] = _i.map(irps).fillna(0).values
            passed_df["stock_rps"] = _c.map(srps).fillna(0).values
        except Exception:
            passed_df["sector_rps"] = 0
            passed_df["stock_rps"] = 0
        logger.info("L3: kept %d triple-resonance stocks (no regular passes)", len(passed_df))

    if passed_df.empty:
        return pd.DataFrame()

    # Add back triple-resonance bypass stocks (with RPS lookup)
    if len(_triple_df) > 0:
        _triple_passed = pd.DataFrame({
            "code": _triple_df["代码"],
            "buy_type": _triple_df["买点类型"],
            "passed": True,
            "reject_reasons": "三级联立豁免",
            "total_score": _triple_df.get("L2_综合得分", 0),
        })
        # Attach RPS data for composite scoring in L4
        if STOCK_RPS.exists() and INDUSTRY_RPS.exists() and INDUSTRY_MAP.exists():
            try:
                srps = pd.read_parquet(STOCK_RPS).sort_values("date").groupby("code").last().reset_index()
                irps = pd.read_parquet(INDUSTRY_RPS).sort_values("date").groupby("industry").last().reset_index()
                imap = pd.read_parquet(INDUSTRY_MAP)
                srps = srps.set_index("code")["rps_20d"]
                irps = irps.set_index("industry")["rps_20d"]
                imap = imap.set_index("code")["industry"]
                _codes = _triple_passed["code"].astype(str).str.zfill(6)
                _industries = _codes.map(imap)
                _triple_passed["sector_rps"] = _industries.map(irps).fillna(0).values
                _triple_passed["stock_rps"] = _codes.map(srps).fillna(0).values
            except Exception:
                _triple_passed["sector_rps"] = 0
                _triple_passed["stock_rps"] = 0
        else:
            _triple_passed["sector_rps"] = 0
            _triple_passed["stock_rps"] = 0
        passed_df = pd.concat([passed_df, _triple_passed], ignore_index=True)
        logger.info("L3: added back %d triple-resonance stocks", len(_triple_passed))

    # Merge L3 filter results back with original L2 fields (exact code+buy_type match)
    passed_df["_key"] = passed_df["code"] + "|" + passed_df["buy_type"].apply(_parse_base_buy_type)
    input_df["_key"] = input_df["代码"] + "|" + input_df["买点类型"].apply(_parse_base_buy_type)
    result = input_df[input_df["_key"].isin(passed_df["_key"])].copy()
    result.drop(columns=["_key"], inplace=True)

    # Attach L3 filter dimensions
    for col in ["sector_rps", "stock_rps", "atr_pct", "vol_ratio", "high_dist"]:
        if col in passed_df.columns:
            lookup = passed_df.drop_duplicates(subset=["code"]).set_index("code")[col]
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
