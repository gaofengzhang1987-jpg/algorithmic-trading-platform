"""从 L2/L3/L4 管道输出提取因子值，对齐前向收益，生成 Alphalens 输入。

因子来源:
  - L2: EntryFilter 各维度原始分 (中枢/量比/MACD/底分型/...)
  - L3: atr_pct, vol_ratio, high_dist, stock_rps, sector_rps
  - L4: composite, n_l2

输出格式: MultiIndex DataFrame (date, asset) x 因子列，符合 Alphalens 要求。
"""
import pandas as pd
import numpy as np
from pathlib import Path

from core.constants import DATA_DIR, SIGNALS_DIR
from core.data import load_daily, load_signals
from core.signal_detector import detect_all_changes
from entry_filter import EntryFilter, REGIME_WEIGHTS


class FactorExtractor:
    """从现有管道提取因子值，对齐前向收益。

    Args:
        regime: BULL / BEAR / CHOP
        lookback_days: 因子提取的回看窗口（默认 500 天）
    """

    def __init__(self, regime="CHOP", lookback_days=500):
        self.regime = regime
        self.lookback_days = lookback_days

    # ── L2 维度名（从 REGIME_WEIGHTS 动态获取） ──
    def _l2_dimension_names(self):
        """汇总所有买点类型在目标 regime 下的维度名。"""
        dims = set()
        for bt in ["一买", "二买", "三买"]:
            w = REGIME_WEIGHTS.get(bt, {}).get(self.regime, {})
            dims.update(w.keys())
        return sorted(dims)

    def extract(self, codes=None, max_stocks=200):
        """提取因子截面数据。

        Returns:
            factor_df: MultiIndex (date, asset) DataFrame, 列为各因子值
            forward_returns: MultiIndex (date, asset) Series, 前向收益
        """
        if codes is None:
            codes = sorted(p.stem for p in SIGNALS_DIR.glob("*.parquet"))
        if max_stocks and len(codes) > max_stocks:
            codes = codes[:max_stocks]

        l2_dims = self._l2_dimension_names()
        # L3/L4 通用维度
        l3l4_dims = ["atr_pct", "vol_ratio", "high_dist", "stock_rps", "sector_rps", "composite", "n_l2"]
        all_factor_cols = l2_dims + l3l4_dims

        records = []
        for code in codes:
            daily = load_daily(code)
            if daily is None or len(daily) < 120:
                continue
            sig_df = load_signals(code)
            if sig_df is None or sig_df.empty:
                continue
            changes = detect_all_changes(sig_df)
            buy_events = [c for c in changes if c["type"] == "buy"]
            if not buy_events:
                continue

            try:
                ef = EntryFilter(code, daily, sig_df, regime=self.regime)
                for buy in buy_events:
                    result = ef.filter(buy)
                    if not result.passed:
                        continue
                    date = pd.Timestamp(buy["date"])
                    row = {"date": date, "asset": code, "buy_type": result.buy_type}
                    # L2 维度分
                    for dim in l2_dims:
                        row[dim] = result.dimension_scores.get(dim, 0.0)
                    # L3/L4 维度暂置 NaN（需运行完整管道才能填充）
                    for dim in l3l4_dims:
                        row[dim] = np.nan
                    records.append(row)
            except Exception:
                continue

        if not records:
            return pd.DataFrame(), pd.Series(dtype=float)

        factor_df = pd.DataFrame(records)
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        factor_df = factor_df.set_index(["date", "asset"]).sort_index()
        # 确保 date level 是 DatetimeIndex（Alphalens 兼容性要求）
        if isinstance(factor_df.index, pd.MultiIndex):
            date_level = factor_df.index.get_level_values("date")
            if not isinstance(date_level, pd.DatetimeIndex):
                factor_df.index = factor_df.index.set_levels(
                    pd.to_datetime(date_level), level="date"
                )

        # 计算前向收益 (1d, 5d, 20d)
        fwd = self._compute_forward_returns(factor_df)

        return factor_df, fwd

    def extract_with_l3l4(self, l4_output_df):
        """从已有的 L4 管道输出提取因子（含 L3/L4 维度）。

        Args:
            l4_output_df: L4Ranker.rank() 返回的 DataFrame
                (需含 atr_pct, vol_ratio, high_dist, stock_rps, sector_rps, composite, n_l2)

        Returns:
            factor_df, forward_returns (同 extract)
        """
        # 需要 date 列；L4 输出只有 code 没有 date，需反查信号日期
        # 从信号文件补充 signal_date
        l4 = l4_output_df.copy()
        dates = []
        for _, row in l4.iterrows():
            code = row["code"]
            sig_df = load_signals(code)
            if sig_df is None or sig_df.empty:
                dates.append(pd.NaT)
                continue
            changes = detect_all_changes(sig_df)
            buys = [c for c in changes if c["type"] == "buy" and
                    c.get("signal_label", "").startswith(row.get("buy_type", ""))]
            dates.append(pd.Timestamp(buys[-1]["date"]) if buys else pd.NaT)

        l4["date"] = dates
        l4 = l4.dropna(subset=["date"])
        l4["date"] = pd.to_datetime(l4["date"])
        l4 = l4.rename(columns={"code": "asset"})

        cols = ["date", "asset", "buy_type", "atr_pct", "vol_ratio", "high_dist",
                "stock_rps", "sector_rps", "composite", "n_l2", "total_score"]
        cols = [c for c in cols if c in l4.columns]
        factor_df = l4[cols].set_index(["date", "asset"]).sort_index()
        # 确保 date level 是 DatetimeIndex（Alphalens 兼容性要求）
        if isinstance(factor_df.index, pd.MultiIndex):
            date_level = factor_df.index.get_level_values("date")
            if not isinstance(date_level, pd.DatetimeIndex):
                factor_df.index = factor_df.index.set_levels(
                    pd.to_datetime(date_level), level="date"
                )
        forward_returns = self._compute_forward_returns(factor_df)
        return factor_df, forward_returns

    def _compute_forward_returns(self, factor_df):
        """对每个因子中的 (date, asset) 计算前向收益。"""
        records = []
        for (dt, code), _ in factor_df.iterrows():
            daily = load_daily(code)
            if daily is None:
                continue
            daily = daily.sort_values("date")
            daily["date"] = pd.to_datetime(daily["date"])
            mask = daily["date"] > dt
            future = daily[mask]
            if len(future) < 20:
                continue
            price_now = daily[daily["date"] <= dt].iloc[-1]["close"]
            for period in (1, 5, 20):
                if len(future) >= period:
                    price_fwd = future.iloc[period - 1]["close"]
                    ret = (price_fwd - price_now) / price_now
                else:
                    ret = np.nan
                records.append({"date": dt, "asset": code,
                                f"{period}d": ret})

        if not records:
            return pd.Series(dtype=float)

        fwd = pd.DataFrame(records).set_index(["date", "asset"]).sort_index()
        return fwd


def _build_prices_for_alphalens(factor_df):
    """从 factor_df 的 (date, asset) 索引构建 Alphalens 兼容的价格矩阵。

    Alphalens 期望 prices 为 DatetimeIndex × asset columns 的 DataFrame。
    """
    import pandas as pd
    from core.data import load_daily

    assets = factor_df.index.get_level_values("asset").unique()
    all_dates = factor_df.index.get_level_values("date").unique()
    price_dict = {}

    for code in assets:
        daily = load_daily(code)
        if daily is None or len(daily) < 60:
            continue
        daily = daily.sort_values("date")
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date")["close"]
        common_dates = daily.index.intersection(all_dates)
        if len(common_dates) > 0:
            price_dict[code] = daily.reindex(all_dates).ffill()

    return pd.DataFrame(price_dict).dropna(axis=1, thresh=3) if price_dict else None
