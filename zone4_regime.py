#!/usr/bin/env python3
"""L4 Regime Ranker — wraps l4_ranker.L4Ranker for pipeline integration."""

import logging
from pathlib import Path
import pandas as pd
from l4_ranker import L4Ranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from qlib_ml.signal_predictor import SignalQlibPredictor
except ImportError:
    SignalQlibPredictor = None

logger = logging.getLogger("zone4_regime")

BASE = Path(__file__).parent
ZONES = BASE / "data" / "zones"
NAME_DB = BASE / "data" / "industry_classification.parquet"


def _parse_base_buy_type(label: str) -> str:
    if "一买" in label: return "一买"
    if "二买" in label: return "二买"
    if "三买" in label: return "三买"
    return "一买"


def run(input_df=None, top_n=100, skip_fundamental: bool = False, signal_date: str = None):
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

    # Triple-resonance: boost sector_rps to bypass sector cull, get real composite
    _triple_mask = input_df["买点类型"].str.contains("30m_结构联立", na=False).values
    _triple_codes = set(l4_input.loc[_triple_mask, "code"])
    if _triple_codes:
        l4_input.loc[_triple_mask, "sector_rps"] = 100.0
        logger.info("L4 Ranker: %d triple-resonance stocks, sector_rps boosted to 100",
                    len(_triple_codes))

    qlib_pred = None
    if SignalQlibPredictor is not None:
        try:
            qlib_pred = SignalQlibPredictor()
        except Exception:
            logger.warning("QlibPredictor failed to load, continuing without ML enhancement")
    ranker = L4Ranker(qlib_predictor=qlib_pred)
    ranked = ranker.rank(l4_input, signal_date=signal_date)

    if ranked.empty:
        logger.info("L4 Ranker: 0 passed after sector cull")
        return pd.DataFrame()

    # Attach price and original label from L3
    # Sort to prioritize triple-resonance entries for label mapping
    _idf_sorted = input_df.copy()
    _idf_sorted["_has_triple"] = _idf_sorted["买点类型"].str.contains("30m_结构联立", na=False)
    _idf_sorted = _idf_sorted.sort_values("_has_triple", ascending=False)
    price_map = _idf_sorted.drop_duplicates(subset=["代码"]).set_index("代码")["现价"].to_dict()
    label_map = _idf_sorted.drop_duplicates(subset=["代码"]).set_index("代码")["买点类型"].to_dict()
    regime_map = _idf_sorted.drop_duplicates(subset=["代码"]).set_index("代码").get("L2_Regime", pd.Series("", index=input_df.index)).to_dict()

    ranked["现价"] = ranked["code"].map(price_map)
    ranked["买点类型"] = ranked["code"].map(label_map)
    ranked["L2_Regime"] = ranked["code"].map(regime_map)
    ranked["L2_综合得分"] = ranked["total_score"]

    # 透传 L1 标识列（按代码去重，与 买点类型/现价 同一口径）
    for flag_col in ("非买点转买点", "当天非买转买"):
        if flag_col in input_df.columns:
            _flag_map = _idf_sorted.drop_duplicates(subset=["代码"]).set_index("代码")[flag_col].to_dict()
            ranked[flag_col] = ranked["code"].map(_flag_map).fillna(False).astype(bool)
        else:
            ranked[flag_col] = False

    # Attach stock name and signal date
    if NAME_DB.exists():
        try:
            _ndf = pd.read_parquet(NAME_DB)
            _nm = _ndf.drop_duplicates(subset=["code"]).set_index("code")["name"].to_dict()
            ranked["名称"] = ranked["code"].map(_nm)
        except Exception:
            ranked["名称"] = ""
    else:
        ranked["名称"] = ""
    # Signal date from input (latest date of the signal)
    _sd = _idf_sorted.drop_duplicates(subset=["代码"]).set_index("代码")["最新日期"].to_dict()
    ranked["信号日期"] = ranked["code"].map(_sd)

    # Dedup: keep max composite per code (already sorted by _zone + composite)

    # Save triple-resonance rows (with real composite) before top_n cut
    if _triple_codes:
        _triple_ranked = ranked[ranked["code"].isin(_triple_codes)].copy()

    ranked = ranked.head(top_n)

    ranked = ranked.drop_duplicates(subset=["code"], keep="first")
    # Re-add any triple-resonance stocks cut by top_n — use saved copy with real scores
    if _triple_codes:
        _missing = _triple_codes - set(ranked["code"])
        if _missing:
            _tr_add = _triple_ranked[_triple_ranked["code"].isin(_missing)].copy()
            _tr_add["tag"] = _tr_add["tag"].apply(lambda t: "三级联立豁免" if t == "" else t)
            ranked = pd.concat([ranked, _tr_add[ranked.columns]], ignore_index=True)
            logger.info("L4 Ranker: restored %d triple-resonance stocks cut by top_n",
                        len(_tr_add))

    # Re-sort by composite for consistent global ranking
    ranked = ranked.sort_values("composite", ascending=False).reset_index(drop=True)
    ranked["global_rank"] = range(1, len(ranked) + 1)

    logger.info("L4 Ranker: %d after dedup", len(ranked))

    # 基本面评分叠加（失败不阻断管道）
    if not skip_fundamental:
        try:
            from fundamental import run_overlay
            ranked = run_overlay(ranked)
        except Exception as e:
            logger.warning("基本面评分跳过: %s", e)

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
