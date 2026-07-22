#!/usr/bin/env python3
"""阈值扫描：6 组 buy_type × regime，固定 Optuna 权重，遍历阈值点。"""
import pandas as pd, numpy as np

df = pd.read_parquet('tmp_out/optuna_bullbear_full.parquet')

WEIGHTS = {
    ("一买", "BEAR"): {"中枢": 0.57, "量比": 0.44, "MACD": 0.29, "底分型": 0.13, "核心验证": 0.10, "底部放量": 0.10},
    ("一买", "BULL"): {"量比": 0.70, "中枢": 0.45, "MACD": 0.39, "核心验证": 0.34, "距离": 0.26},
    ("二买", "BEAR"): {"MACD": 1.52, "量比": 0.90, "底分型": 0.44, "核心验证": 0.40},
    ("二买", "BULL"): {"MA防守": 2.96, "MACD": 0.99, "笔数": 0.98, "底分型": 0.08},
    ("三买", "BEAR"): {"笔数": 1.95, "量比": 0.96, "MACD": 0.59, "核心验证": 0.20, "底分型": 0.10},
    ("三买", "BULL"): {"MA排列强度": 1.09, "笔数": 0.90, "MACD": 0.72},
}

SCAN = {
    ("一买", "BEAR"): [60, 65, 70, 75, 80, 87, 95, 101],
    ("一买", "BULL"): [97, 109, 117, 122, 129, 135, 144, 151],
    ("二买", "BEAR"): [164, 168, 178, 182, 187, 191, 196, 201],
    ("二买", "BULL"): [370, 400, 408, 412, 416, 418, 419, 421],
    ("三买", "BEAR"): [59, 65, 68, 70, 75, 83, 86, 90],
    ("三买", "BULL"): [118, 124, 157, 190],
}

for (bt, regime), w in WEIGHTS.items():
    data = df[(df["buy_type"] == bt) & (df["regime"] == regime)].copy()
    true_n = (data["label"] == 1).sum()
    total_n = len(data)

    data["total"] = 0.0
    for d, weight in w.items():
        if d in data.columns:
            data["total"] += data[d].fillna(0) * weight

    print(f"\n{bt} {regime} (n={total_n:,} true={true_n:,})  weights: {w}")
    print(f"  {'阈值':>6}  {'通过':>8}  {'通过率':>7}  {'Prec':>6}  {'Recall':>7}")
    print("  " + "-" * 45)

    for th in SCAN.get((bt, regime), []):
        passed = data[data["total"] >= th]
        if len(passed) == 0:
            continue
        tp = (passed["label"] == 1).sum()
        prec = tp / len(passed)
        rec = tp / true_n
        print(f"  {th:>6}  {len(passed):>8,}  {len(passed)/total_n:.1%}    {prec:.3f}  {rec:.3f}")
