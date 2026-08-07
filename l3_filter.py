#!/usr/bin/env python3
"""L3 Filter: Regime Routing + Priority Dedup + 5-Day Shield.

Supports BULL, BEAR, CHOP with per-regime thresholds.
Input: L2 EntryFilter candidates (DataFrame with code, buy_type, signal_date, total_score)
Output: L3 candidates (DataFrame with passed, reject_reasons, dimensions)
"""
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timedelta

WORKDIR = Path("/Users/hz/Desktop/Algorithmic Trading Platform")
DAILY = WORKDIR / "data" / "daily"
STOCK_RPS = WORKDIR / "data" / "reference" / "stock_rps.parquet"
INDUSTRY_RPS = WORKDIR / "data" / "reference" / "industry_rps.parquet"
INDUSTRY_MAP = WORKDIR / "data" / "industry_classification.parquet"

THRESHOLDS = {
    "BULL": {
        "freshness_days": 30,
        "weekly_trend": "close>ma10",
        "vol_ratio": 0.8,
        "atr_pct": 0.07,
        "high_dist_buy1": -0.08,
        "high_dist_buy23": -0.08,
        "daily_trend": False,
        "level2_enabled": False,
    },
    "BEAR": {
        "freshness_days": 30,
        "weekly_trend": "slope_up_close>ma10_0.90",
        "vol_ratio": 1.5,
        "atr_pct": 0.04,
        "high_dist_buy1": -0.03,
        "high_dist_buy23": -0.03,
        "daily_trend": True,
        "level2_enabled": False,
    },
    "CHOP": {
        "freshness_days": 30,
        "weekly_trend": "close>ma10_0.95",
        "vol_ratio": 1.0,
        "atr_pct": 0.06,
        "high_dist_buy1": -0.10,
        "high_dist_buy23": -0.10,
        "daily_trend": False,
        "level2_enabled": False,  # CHOP 关闭 Level2 质量过滤
    },
}


class L3Filter:
    """L3 signal availability filter with regime routing."""

    _rps_stock = None
    _rps_industry = None
    _industry_map = None

    @classmethod
    def _load_rps(cls):
        """Lazy-load RPS data (shared across all instances)."""
        if cls._rps_stock is None and STOCK_RPS.exists():
            sr = pd.read_parquet(STOCK_RPS)
            sr_latest = sr[sr["date"] == sr["date"].max()].set_index("code")
            cls._rps_stock = sr_latest
        if cls._rps_industry is None and INDUSTRY_RPS.exists():
            ir = pd.read_parquet(INDUSTRY_RPS)
            ir_latest = ir[ir["date"] == ir["date"].max()].set_index("industry")
            cls._rps_industry = ir_latest
        if cls._industry_map is None and INDUSTRY_MAP.exists():
            cls._industry_map = pd.read_parquet(INDUSTRY_MAP).set_index("code")

    def __init__(self, regime="CHOP"):
        self.regime = regime
        self.thr = THRESHOLDS[regime]
        self._recent_passes = {}  # code -> last pass date (for 5-day shield)

    def filter_batch(self, l2_df):
        """Filter a batch of L2-passed candidates.

        Args:
            l2_df: DataFrame with columns [code, buy_type, signal_date, total_score, ...]
        Returns:
            DataFrame with columns [code, buy_type, passed, reject_reasons,
                                    atr_pct, vol_ratio, high_dist, total_score]
        """
        # Priority dedup: 一买 > 二买 > 三买 (keep highest priority per code)
        priority = {"一买": 3, "二买": 2, "三买": 1}
        l2_df = l2_df.copy()
        l2_df["_pri"] = l2_df["buy_type"].map(priority)
        l2_df = l2_df.sort_values("_pri", ascending=False)
        l2_df = l2_df.drop_duplicates("code", keep="first")
        l2_df = l2_df.drop(columns=["_pri"])

        results = []
        for _, row in l2_df.iterrows():
            code = row["code"]
            # 5-day shield: skip if this code passed L3 within last 5 calendar days
            last_pass = self._recent_passes.get(code)
            if last_pass and (datetime.now() - last_pass).days < 5:
                results.append({
                    "code": code, "buy_type": row["buy_type"],
                    "passed": False, "reject_reasons": "5天屏蔽",
                    "atr_pct": 0, "vol_ratio": 0, "high_dist": 0,
                    "total_score": row.get("total_score", 0),
                })
                continue

            result = self._filter_one(row)
            if result["passed"]:
                self._recent_passes[code] = datetime.now()
            results.append(result)

        result_df = pd.DataFrame(results)
        # Level 3: attach RPS labels (identification only, not scored)
        self._attach_rps(result_df)
        return result_df

    def _attach_rps(self, df):
        """Attach sector/stock RPS as Level 3 identification labels.
        Does NOT affect pass/fail decision.
        """
        self._load_rps()
        sector_vals, stock_vals = [], []
        for _, row in df.iterrows():
            code = row["code"]
            # Stock RPS
            if self._rps_stock is not None and code in self._rps_stock.index:
                val = self._rps_stock.loc[code, "rps_20d"]
                stock_vals.append(float(val) if pd.notna(val) else 0.0)
            else:
                stock_vals.append(0.0)
            # Sector RPS (via industry classification)
            s_val = 0.0
            if self._industry_map is not None and code in self._industry_map.index:
                ind = self._industry_map.loc[code, "industry"]
                if isinstance(ind, pd.Series):
                    ind = ind.iloc[0]
                if self._rps_industry is not None and ind in self._rps_industry.index:
                    s_val = float(self._rps_industry.loc[ind, "rps_20d"])
            sector_vals.append(s_val)
        df["sector_rps"] = sector_vals
        df["stock_rps"] = stock_vals


    def _filter_one(self, row):
        """Filter a single L2 candidate through L3 dimensions."""
        code = row["code"]
        buy_type = row["buy_type"]
        signal_date = pd.Timestamp(row.get("signal_date", row.get("dt", "2026-01-01")))
        total_score = row.get("total_score", 0)
        reasons = []
        atr_pct = 0.0
        high_dist = 0.0
        vol_ratio = 0.0

        try:
            dd = pd.read_parquet(DAILY / f"{code}.parquet")
        except Exception:
            return self._fail(code, buy_type, "数据读取失败", total_score)

        if len(dd) < 60:
            return self._fail(code, buy_type, "数据不足(＜60日)", total_score)

        dd = dd.sort_values("date").tail(250).reset_index(drop=True)
        dd["date"] = pd.to_datetime(dd["date"])
        dd["ma20"] = dd["close"].rolling(20).mean()
        last = dd.iloc[-1]

        # 1. Signal freshness
        if (dd["date"].max() - signal_date).days > self.thr["freshness_days"]:
            reasons.append("信号过期")

        # Level 2: Quality filters (may be disabled per regime)
        if self.thr.get("level2_enabled", True):
            # 2. Daily trend (BEAR only)
            if self.thr.get("daily_trend", False):
                if pd.notna(last["ma20"]) and last["close"] <= last["ma20"]:
                    reasons.append("日线趋势不和")

        # 3. Weekly trend
        w_ok = self._check_weekly(dd, buy_type)
        if not w_ok:
            reasons.append("周线趋势不和")

        # BEAR 专项检查 (2026-07-31 新增)
        if self.regime == "BEAR":
            # 4. 5日动量防线
            if len(dd) >= 6:
                roc_5d = (last["close"] - dd.iloc[-6]["close"]) / dd.iloc[-6]["close"] * 100
                if roc_5d < 0:
                    reasons.append("5日动量不足")
            # 5. 60日跌幅区间
            if len(dd) >= 60:
                high60 = dd["high"].tail(60).max()
                decline_60d = (last["close"] - high60) / high60 * 100
                if buy_type == "一买":
                    if decline_60d > -10 or decline_60d < -50:
                        reasons.append(f"60日跌幅不合({decline_60d:.0f}%)")
                elif buy_type == "二买":
                    if decline_60d > -3 or decline_60d < -30:
                        reasons.append(f"60日跌幅不合({decline_60d:.0f}%)")
                elif buy_type == "三买":
                    if decline_60d > -3 or decline_60d < -40:
                        reasons.append(f"60日跌幅不合({decline_60d:.0f}%)")


        # Level 2: Volume / ATR / High distance
        if self.thr.get("level2_enabled", True):
            # 4. Volume confirmation
            vol_col = "vol" if "vol" in dd.columns else "volume"
            vol_ratio = 0
            if vol_col in dd.columns:
                vol_ratio = last[vol_col] / dd[vol_col].tail(20).mean()
                if vol_ratio < self.thr.get("vol_ratio", 1.0):
                    reasons.append("放量不足")

            # 5. ATR filter
            h, l, c = dd["high"], dd["low"], dd["close"]
            tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
            ).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr14 / last["close"]
            if atr_pct >= self.thr.get("atr_pct", 1.0):
                reasons.append(f"ATR过大({atr_pct*100:.0f}%)")

            # 6. Distance from 20-day high
            high20 = dd["high"].tail(20).max()
            high_dist = (last["close"] - high20) / last["close"]
            if buy_type == "一买":
                limit = self.thr.get("high_dist_buy1")
            else:
                limit = self.thr.get("high_dist_buy23")
            if limit is not None and high_dist < limit:
                reasons.append(f"距前高太远({high_dist*100:.0f}%)")
            else:
                atr_pct = 0
                high_dist = 0

        passed = len(reasons) == 0
        return {
            "code": code,
            "buy_type": buy_type,
            "passed": passed,
            "reject_reasons": ", ".join(reasons) if reasons else "",
            "atr_pct": round(atr_pct * 100, 2),
            "vol_ratio": round(vol_ratio, 2),
            "high_dist": round(high_dist * 100, 2),
            "total_score": total_score,
        }

    def _check_weekly(self, dd, buy_type=""):
        """Check weekly trend based on regime."""
        try:
            weekly = (
                dd.set_index("date")
                .resample("W")
                .agg({"close": "last"})
                .dropna()
            )
            if len(weekly) < 10:
                return True  # insufficient data, pass
            weekly["ma10"] = weekly["close"].rolling(10).mean()
            w_last = weekly.iloc[-1]
            if pd.isna(w_last["ma10"]):
                return True

            trend_type = self.thr["weekly_trend"]

            if trend_type == "close>ma10":
                return w_last["close"] > w_last["ma10"]

            elif trend_type == "close>ma10_0.95":
                return w_last["close"] > w_last["ma10"] * 0.95

            elif trend_type == "slope_up_close>ma10_0.90":
                # BEAR: close > MA10 × 0.90 + 4w斜率 ≥ −2 (2026-07-31更新)
                if not (w_last["close"] > w_last["ma10"] * 0.90):
                    return False
                # 一买豁免斜率检查
                if "一买" in buy_type:
                    return True
                if len(weekly) < 4:
                    return True
                slope = (
                    (weekly["ma10"].iloc[-1] - weekly["ma10"].iloc[-4])
                    / weekly["ma10"].iloc[-4]
                    * 100
                )
                return slope >= -2

            return True
        except Exception:
            return True  # error, pass

    def _fail(self, code, buy_type, reason, total_score=0):
        return {
            "code": code,
            "buy_type": buy_type,
            "passed": False,
            "reject_reasons": reason,
            "atr_pct": 0,
            "vol_ratio": 0,
            "high_dist": 0,
            "total_score": total_score,
        }
