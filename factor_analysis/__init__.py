"""因子分析模块 — Alphalens 集成。

用法:
    from factor_analysis import FactorExtractor, run_alphalens_analysis

    # 从 L2/L3/L4 管道输出提取因子
    extractor = FactorExtractor(regime="CHOP")
    factor_df = extractor.extract(codes=["000001", "000002"])

    # 运行 Alphalens 分析
    report = run_alphalens_analysis(factor_df, periods=(1, 5, 20))
"""
from factor_analysis.factor_extractor import FactorExtractor
from factor_analysis.alphalens_runner import run_alphalens_analysis
