"""组合优化模块 — PyPortfolioOpt 集成。

用法:
    from portfolio import PortfolioOptimizer

    opt = PortfolioOptimizer(lookback=120)
    weights = opt.optimize(candidates_df)
    # weights: dict[code -> weight_pct], sum = 1.0
"""
from portfolio.optimizer import PortfolioOptimizer
