"""ML 增强预测模块 — LightGBM 因子训练 + 预测。

用法:
    from qlib_ml import train, QlibPredictor

    # 训练模型
    train(horizon=20, max_stocks=300)

    # 预测
    pred = QlibPredictor()
    scores = pred.score(["000001", "000002"], signal_date="2026-08-04")
"""
from qlib_ml.trainer import train
from qlib_ml.predictor import QlibPredictor
