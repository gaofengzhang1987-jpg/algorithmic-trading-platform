"""组合优化器 — 在 L4 排序后对候选池做仓位优化。

支持方法:
  - HRP (Hierarchical Risk Parity): 默认，适合 A 股板块联动
  - Risk Parity: 等风险贡献
  - Max Sharpe: 最大夏普（需预期收益估计）
  - Equal Weight: 等权（基准对照）

关键假设:
  - 候选池 ≤30 只（L4 排名 + sector cull 后通常 10-20 只）
  - 协方差矩阵基于过去 60-120 个交易日日收益
  - 优化后权重直接映射到 backtest CAPITAL_PER_TRADE 分配
"""
import pandas as pd
import numpy as np
from pathlib import Path

from core.constants import DATA_DIR, CAPITAL_PER_TRADE
from core.data import load_daily
from pypfopt import risk_models, expected_returns, HRPOpt, EfficientFrontier
import scipy.optimize as sco


class PortfolioOptimizer:
    """L4 候选池的组合仓位优化。

    Args:
        lookback: 协方差估计的回看天数（默认 120）
        method: 优化方法 ("hrp" | "risk_parity" | "max_sharpe" | "equal_weight")
        max_weight: 单只股票最大权重上限
        min_weight: 单只股票最小权重下限（低于此会被剔除）
    """

    def __init__(self, lookback=120, method="hrp",
                 max_weight=0.25, min_weight=0.02):
        self.lookback = lookback
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize(self, candidates_df, signal_date=None):
        """对候选池做仓位优化。

        Args:
            candidates_df: L4Ranker.rank() 输出, 必须含 "code" 列
            signal_date: 信号日期，用于截断价格数据（取该日之前的 lookback 天）

        Returns:
            dict: {code: weight} 权重映射，sum ≈ 1.0
        """
        codes = candidates_df["code"].tolist()
        if not codes:
            return {}
        # 单只股票：100% 赋权
        if len(codes) == 1:
            return {codes[0]: 1.0}
        # Step 1: 构建价格矩阵 (lookback × N)
        price_matrix = self._build_price_matrix(codes, signal_date)
        if price_matrix is None or price_matrix.shape[1] < 2:
            return self._equal_weights(codes)
        # Step 2: 计算收益 + 协方差
        returns = price_matrix.pct_change().dropna()
        if len(returns) < 20:
            return self._equal_weights(codes)
        cov = risk_models.sample_cov(price_matrix, frequency=252)
        mu = expected_returns.mean_historical_return(price_matrix, frequency=252)
        # Step 3: 过滤异常协方差（NaN/inf → 剔除对应股票）
        valid_mask = ~(cov.isna().any(axis=0) | cov.isna().any(axis=1))
        valid_codes = [c for c in codes if c in valid_mask.index and valid_mask[c]]
        if len(valid_codes) < 2:
            return self._equal_weights(codes)
        # 重裁有效股票
        cov = cov.loc[valid_codes, valid_codes]
        mu = mu[valid_codes]
        # Step 4: 执行优化
        try:
            weights = self._run_optimization(mu, cov, price_matrix[valid_codes])
        except Exception:
            weights = self._equal_weights(valid_codes)
        # Step 5: 归一化 + 下限裁剪
        return self._normalize_and_clip(weights, valid_codes)

    # ── Private ──

    def _build_price_matrix(self, codes, signal_date):
        """从日线数据构建价格矩阵。"""
        price_dict = {}
        for code in codes:
            daily = load_daily(code)
            if daily is None or len(daily) < self.lookback:
                continue
            daily = daily.sort_values("date")
            daily["date"] = pd.to_datetime(daily["date"])
            if signal_date is not None:
                daily = daily[daily["date"] <= pd.Timestamp(signal_date)]
            daily = daily.tail(self.lookback)
            if len(daily) < 20:
                continue
            price_dict[code] = daily.set_index("date")["close"]
        if not price_dict:
            return None
        df = pd.DataFrame(price_dict).dropna(axis=1)
        if df.shape[1] < 2:
            return None
        return df

    def _run_optimization(self, mu, cov, prices):
        """执行选定的优化方法。"""
        codes = cov.columns.tolist()
        if self.method == "hrp":
            hrp = HRPOpt(prices.pct_change().dropna())
            hrp_weights = hrp.optimize()
            return {c: hrp_weights.get(c, 0) for c in codes}
        if self.method == "risk_parity":
            n = len(codes)
            def rp_objective(w):
                w_abs = np.abs(w) / np.sum(np.abs(w))
                port_vol = np.sqrt(w_abs @ cov.values @ w_abs)
                mrc = (cov.values @ w_abs) / port_vol
                rc = w_abs * mrc
                target = port_vol / n
                return np.sum((rc - target) ** 2)
            init = np.ones(n) / n
            bounds = [(0, self.max_weight)] * n
            result = sco.minimize(rp_objective, init, bounds=bounds, method="SLSQP")
            raw = np.abs(result.x)
            total = raw.sum()
            return {c: float(raw[i] / total) for i, c in enumerate(codes)}
        if self.method == "max_sharpe":
            ef = EfficientFrontier(mu, cov, weight_bounds=(0, self.max_weight))
            w = ef.max_sharpe(risk_free_rate=0.02)
            w = ef.clean_weights()
            return {c: w.get(c, 0) for c in codes}
        # fallback: equal weight
        return self._equal_weights(codes)

    def _normalize_and_clip(self, raw_weights, codes):
        """归一化 + 下限裁剪 + 重新归一化。"""
        w = {c: max(raw_weights.get(c, 0), 0) for c in codes}
        total = sum(w.values())
        if total <= 0:
            return self._equal_weights(codes)
        # 归一化
        w = {c: v / total for c, v in w.items()}
        # 下限裁剪：低于 min_weight 的剔除
        below = [c for c, v in w.items() if v < self.min_weight]
        for c in below:
            del w[c]
        if not w:
            return self._equal_weights(codes)
        # 重新归一化
        total = sum(w.values())
        w = {c: v / total for c, v in w.items()}
        return w

    @staticmethod
    def _equal_weights(codes):
        n = len(codes)
        return {c: 1.0 / n for c in codes}

    def allocate_capital(self, weights, total_capital=CAPITAL_PER_TRADE):
        """将权重转为金额分配。

        Args:
            weights: dict[code -> weight]
            total_capital: 总资金（默认从 constants 读取）
        Returns:
            dict: {code: allocated_amount}
        """
        return {code: round(total_capital * w, 2) for code, w in weights.items()}
