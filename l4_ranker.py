"""L4 Ranker: Sector cull + within-type L2 normalization + composite + zone output."""
import pandas as pd, numpy as np


class L4Ranker:
    """L4 ranker with per-buy-type normalization and zoned output.

    Fix: L2 total_score is normalized within each buy_type group,
    so 三买 high-scale scores don't crush low-scale 二买/一买.

    Output: grouped by buy_type (三买 ★★★ first in CHOP), ranked within each zone.
    """

    def __init__(self, sector_rps_min: float = 40.0,
                 w_l2: float = 0.40, w_stock_rps: float = 0.40, w_sector_rps: float = 0.20):
        self.sector_rps_min = sector_rps_min
        self.w_l2 = w_l2
        self.w_stock_rps = w_stock_rps
        self.w_sector_rps = w_sector_rps

    def rank(self, l3_df):
        """Rank L3 candidates with per-type L2 normalization.

        Returns DataFrame grouped by buy_type (三买→二买→一买),
        ranked within each zone by composite score.
        """
        df = l3_df[l3_df["passed"]].copy()
        if df.empty:
            return df

        # Step 1: sector cull
        rejected = df[df["sector_rps"] < self.sector_rps_min]
        df = df[df["sector_rps"] >= self.sector_rps_min]
        if df.empty:
            return df

        # Step 2: per-buy-type L2 normalization
        df["n_l2"] = 0.0
        for bt in df["buy_type"].unique():
            mask = df["buy_type"] == bt
            scores = df.loc[mask, "total_score"]
            smin, smax = scores.min(), scores.max()
            if smax > smin:
                df.loc[mask, "n_l2"] = (scores - smin) / (smax - smin)
            else:
                df.loc[mask, "n_l2"] = 0.5

        # Step 3: global normalization for RPS dimensions
        def _norm(series):
            vmin, vmax = series.min(), series.max()
            if vmax == vmin:
                return pd.Series(0.5, index=series.index)
            return (series - vmin) / (vmax - vmin)

        n_sr = _norm(df["stock_rps"])
        n_ir = _norm(df["sector_rps"])

        composite = (
            self.w_l2 * df["n_l2"]
            + self.w_stock_rps * n_sr
            + self.w_sector_rps * n_ir
        )
        df["composite"] = composite.round(4)

        # Step 4: zoned ranking (三买 first, then 二买, then 一买)
        buy_order = {"三买": 0, "二买": 1, "一买": 2}
        df["_zone"] = df["buy_type"].map(buy_order)
        df = df.sort_values(["_zone", "composite"], ascending=[True, False])
        df = df.drop(columns=["_zone"])

        # Step 5: per-zone rank + inter-zone rank
        df["zone_rank"] = 0
        df["global_rank"] = 0
        g_rank = 0
        for bt in ["三买", "二买", "一买"]:
            mask = df["buy_type"] == bt
            if mask.sum() == 0:
                continue
            indices = df[mask].index
            n = len(indices)
            df.loc[indices, "zone_rank"] = range(1, n + 1)
            for i, idx in enumerate(indices):
                g_rank += 1
                df.loc[idx, "global_rank"] = g_rank

        # Mark 三买 in CHOP
        df["tag"] = ""
        df.loc[df["buy_type"] == "三买", "tag"] = "★★★"

        # Attach metadata
        df.attrs["rejected_by_sector"] = rejected["code"].tolist() if len(rejected) > 0 else []
        df.attrs["total_l3"] = len(l3_df[l3_df["passed"]])
        df.attrs["after_sector_cull"] = len(df)

        return df.reset_index(drop=True)
