"""信号质量分类器 —— 预测 czsc 买点信号出现后 20 日内能否盈利 5%+。

与 trainer.py 的区别：预测对象从"任意一天的未来收益"改为"买点信号的质量"。
标签与 L4 使用场景完全对齐。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json, sys, warnings
from pathlib import Path as _Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from core.constants import DATA_DIR, BASE_DIR

SIGNAL_DIR = BASE_DIR / "data" / "signals"
MODEL_DIR = BASE_DIR / "qlib_ml" / "models"

# ── 超参 ──
PROFIT_TARGET = 0.05      # 盈利阈值
FORWARD_DAYS = 20          # 前看天数
MIN_BARS_BEFORE = 120      # 信号前需要的最少 bar 数（用于特征计算）
MIN_FORWARD_BARS = 10      # 信号后最少需要的数据天数（不够则丢弃）

# 时间分割
TRAIN_END = pd.Timestamp("2025-12-31")
VALID_END = pd.Timestamp("2026-07-30")

# 合成指数 Regime 缓存
_regime_map = None


def _load_regime():
    global _regime_map
    if _regime_map is not None:
        return _regime_map

    from regime_detector import _compute_bull_score_for_row, _classify, _compute_mas

    base = BASE_DIR / "data" / "index"
    idx = pd.read_parquet(base / "000001.parquet").sort_values("date")
    idx["date"] = pd.to_datetime(idx["date"])
    idx2 = pd.read_parquet(base / "000852.parquet").sort_values("date")
    idx2["date"] = pd.to_datetime(idx2["date"])
    idx = _compute_mas(idx)
    idx2 = _compute_mas(idx2)

    _regime_map = {}
    for i in range(60, len(idx)):
        d = idx["date"].iloc[i]
        score = _compute_bull_score_for_row(idx, idx2, i)
        regime = _classify(score)
        _regime_map[pd.Timestamp(d)] = regime

    return _regime_map
    si["date"] = pd.to_datetime(si["date"])
    si = si.sort_values("date")
    si["ma20"] = si["close"].rolling(20).mean()
    si["ma60"] = si["close"].rolling(60).mean()

    def _r(row):
        if pd.isna(row["ma20"]) or pd.isna(row["ma60"]):
            return "CHOP"
        if row["close"] > row["ma20"] > row["ma60"]:
            return "BULL"
        if row["close"] < row["ma20"] < row["ma60"]:
            return "BEAR"
        return "CHOP"

    si["regime"] = si.apply(_r, axis=1)
    _regime_map = dict(zip(si["date"], si["regime"]))
    return _regime_map


def _parse_buy_type(raw):
    """从信号值（如 '一买_7笔_任意_0'）解析买点类型。"""
    raw = str(raw)
    for bt in ["一买", "二买", "三买"]:
        if bt in raw:
            return bt
    return None


def _parse_stroke_count(raw):
    """从信号值解析笔数（如 '一买_7笔_任意_0' → 7）。"""
    import re
    m = re.search(r"(\d+)笔", str(raw))
    return int(m.group(1)) if m else 0


def _extract_features(daily, signal_idx, regime_map):
    """从日线数据和信号位置提取 ~20 维特征。

    Args:
        daily: 该股票的完整日线 DataFrame（已按 date 排序）
        signal_idx: 信号在 daily 中的行索引
        regime_map: {date: regime} 字典
        
        cluster_size: 当前信号块连续天数
        days_since_last: 距上一信号块的天数
        stroke_count: 信号笔数
    Returns:
        list of float, 长度 26
    """
    i = signal_idx
    c = daily["close"].values.astype(float)
    h = daily["high"].values.astype(float)
    l = daily["low"].values.astype(float)
    v = daily["volume"].values.astype(float)
    o = daily["open"].values.astype(float)
    dates = daily["date"].values
    n = len(c)

    feats = []

    # 1. 收益动量 (3)
    for lag in [5, 10, 20]:
        feats.append(c[i] / c[i - lag] - 1 if i >= lag else 0)

    # 2. MA 偏离 (3)
    for w in [5, 20, 60]:
        ma = np.mean(c[max(0, i - w + 1):i + 1])
        feats.append(c[i] / ma - 1 if ma > 0 else 0)

    # 3. 波动率 (2)
    for w in [10, 20]:
        feats.append(np.std(c[max(0, i - w + 1):i + 1]) / c[i] if c[i] > 0 else 0)

    # 4. ATR 比率 (1)
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    atr14 = np.mean(tr[max(0, i - 13):i + 1])
    feats.append(atr14 / c[i] if c[i] > 0 else 0)

    # 5. 高低价位置 (2)
    h60 = np.max(h[max(0, i - 59):i + 1])
    l60 = np.min(l[max(0, i - 59):i + 1])
    feats.append(c[i] / h60 - 1 if h60 > 0 else 0)
    feats.append((c[i] - l60) / (h60 - l60 + 1e-9))

    # 6. 量价关系 (4)
    v20 = np.mean(v[max(0, i - 19):i + 1])
    feats.append(v[i] / (v20 + 1e-9) - 1)                          # 当日量比
    v5m = np.mean(v[max(0, i - 4):i + 1])
    v20m = np.mean(v[max(0, i - 19):i + 1])
    feats.append(v5m / (v20m + 1e-9) - 1)                          # 短期均量变化
    # 价量相关
    w = min(i + 1, 10)
    pv = np.corrcoef(c[i - w + 1:i + 1], np.log(v[i - w + 1:i + 1] + 1))[0, 1]
    feats.append(pv if not np.isnan(pv) else 0)
    feats.append(1.0 if c[i] > o[i] and v[i] > v20 else 0.0)       # 放量阳线

    # 7. 趋势强度 (2)
    for w in [10, 20]:
        diffs = np.diff(c[max(0, i - w):i + 1])
        feats.append(np.sum(diffs > 0) / max(len(diffs), 1))

    # 8. Regime (3) — one-hot BULL/BEAR/CHOP
    sig_date = pd.Timestamp(dates[i])
    regime = regime_map.get(sig_date, "CHOP")
    feats.append(1.0 if regime == "BULL" else 0.0)
    feats.append(1.0 if regime == "BEAR" else 0.0)
    feats.append(1.0 if regime == "CHOP" else 0.0)

    # 补齐/截断到 20 维
    while len(feats) < 20:
        feats.append(0.0)
    return [float(v) if not np.isnan(v) else 0.0 for v in feats[:20]]


def _compute_label(daily, signal_idx):
    """计算标签：未来 FORWARD_DAYS 内 max_return 是否 >= PROFIT_TARGET。"""
    c = daily["close"].values.astype(float)
    i = signal_idx
    end = min(len(c), i + 1 + FORWARD_DAYS)
    if end <= i + MIN_FORWARD_BARS:
        return None  # 数据不足
    fwd_max = np.max(c[i + 1:end])
    return int(fwd_max / c[i] - 1 >= PROFIT_TARGET)


def build_dataset():
    """从信号文件和日线数据构建训练数据集。

    Returns:
        X: np.ndarray (n_samples, 20)
        y: np.ndarray (n_samples,)
        meta: DataFrame with code, date, buy_type, label
    """
    regime_map = _load_regime()
    signal_files = sorted(Path(SIGNAL_DIR).glob("*.parquet"))
    buy_col = "日线_D1B_BUY1"

    all_X, all_y, all_meta = [], [], []
    errors = 0

    for fi, sf in enumerate(signal_files):
        code = sf.stem
        try:
            sdf = pd.read_parquet(sf)
            ddf = pd.read_parquet(Path(DATA_DIR) / f"{code}.parquet")
        except Exception:
            errors += 1
            continue

        ddf = ddf.sort_values("date").reset_index(drop=True)
        ddf["date"] = pd.to_datetime(ddf["date"])
        if len(ddf) < MIN_BARS_BEFORE:
            continue

        # 只取有实际买点信号的行，且去重连续同类型
        sdf["_buy_type"] = sdf[buy_col].apply(_parse_buy_type)
        sdf["_is_signal"] = sdf["_buy_type"].notna()
        sdf["dt"] = pd.to_datetime(sdf["dt"])

        # 去重：连续同类型只保留第一天
        prev_bt = None
        signal_rows = []
        for idx, row in sdf.iterrows():
            bt = row["_buy_type"]
            if bt is None:
                prev_bt = None
                continue
            if bt == prev_bt:
                continue  # 跳过后面的连续同类型
            prev_bt = bt
            signal_rows.append(row)

        for row in signal_rows:
            sig_date = pd.Timestamp(row["dt"])
            # 在日线数据中定位信号日期
            mask = ddf["date"] == sig_date
            if not mask.any():
                continue
            sig_idx = ddf.index[mask][0]

            if sig_idx < MIN_BARS_BEFORE:
                continue

            label = _compute_label(ddf, sig_idx)
            if label is None:
                continue

            feats = _extract_features(ddf, sig_idx, regime_map)
            all_X.append(feats)
            all_y.append(label)
            all_meta.append({
                "code": code, "date": sig_date,
                "buy_type": row["_buy_type"],
                "stroke_count": _parse_stroke_count(row[buy_col]),
                "label": label,
            })

        if (fi + 1) % 200 == 0:
            print(f"  进度: {fi + 1}/{len(signal_files)}, 样本: {len(all_X)}, 错误: {errors}")

    print(f"总样本: {len(all_X)}, 错误股票: {errors}")
    print(f"正样本(>=5%): {sum(all_y)} ({sum(all_y) / len(all_y) * 100:.1f}%)")

    meta = pd.DataFrame(all_meta)
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.int32), meta


def train():
    """训练信号质量分类器。"""
    import lightgbm as lgb

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("构建信号质量数据集...")
    X, y, meta = build_dataset()

    meta["date"] = pd.to_datetime(meta["date"])
    train_mask = meta["date"] <= TRAIN_END
    valid_mask = (meta["date"] > TRAIN_END) & (meta["date"] <= VALID_END)

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]

    print(f"训练集: {len(X_train)}（正样本 {y_train.sum()} / {y_train.mean()*100:.1f}%）")
    print(f"验证集: {len(X_valid)}（正样本 {y_valid.sum()} / {y_valid.mean()*100:.1f}%）")

    if len(X_train) < 500:
        raise RuntimeError(f"训练集不足: {len(X_train)}")

    # 按 regime 拆分验证集
    for regime in ["BULL", "BEAR", "CHOP"]:
        mask = np.array([_load_regime().get(d, "CHOP") == regime for d in meta["date"][valid_mask]])
        if mask.sum() > 0:
            print(f"  验证集 {regime}: {mask.sum()} 条, 正样本率 {y_valid[mask].mean()*100:.1f}%")

    # ── 训练 ──
    print("训练 LightGBM 分类器...")
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "min_data_in_leaf": 30,
        "is_unbalance": True,
        "verbose": -1,
        "num_threads": 4,
    }

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)],
    )

    model_path = MODEL_DIR / "lgb_model_signal.txt"
    model.save_model(str(model_path))
    print(f"模型已保存: {model_path}")

    # ── 评估 ──
    y_prob = model.predict(X_valid)
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    auc = roc_auc_score(y_valid, y_prob)
    print(f"验证集 AUC: {auc:.4f}")

    # Precision@top-20%
    top_k = max(1, int(len(y_valid) * 0.2))
    top_idx = np.argsort(y_prob)[-top_k:]
    prec_top = y_valid[top_idx].mean()
    print(f"Precision@top20%: {prec_top:.4f}")

    # 特征重要性
    importance = model.feature_importance(importance_type="gain")
    feat_names = [
        "ret5d", "ret10d", "ret20d",
        "ma5_dev", "ma20_dev", "ma60_dev",
        "vol10d", "vol20d",
        "atr_ratio",
        "dist_60h", "pos_60hl",
        "vol_ratio", "vol_trend", "pv_corr", "up_vol_flag",
        "up_days10", "up_days20",
        "regime_BULL", "regime_BEAR", "regime_CHOP",
    ]
    imp_df = pd.DataFrame({"feature": feat_names[:20], "gain": importance[:20]})
    imp_df = imp_df.sort_values("gain", ascending=False)
    print("\n特征重要性 (gain):")
    for _, row in imp_df.head(10).iterrows():
        print(f"  {row['feature']:15s}: {row['gain']:.0f}")

    # 保存配置
    config = {
        "train_end": str(TRAIN_END.date()),
        "valid_end": str(VALID_END.date()),
        "profit_target": PROFIT_TARGET,
        "forward_days": FORWARD_DAYS,
        "n_features": X_train.shape[1],
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "train_pos_rate": float(y_train.mean()),
        "valid_pos_rate": float(y_valid.mean()),
        "auc": round(float(auc), 4),
        "prec_top20": round(float(prec_top), 4),
        "created_at": datetime.now().isoformat(),
    }
    from sklearn.metrics import roc_auc_score
    config_path = MODEL_DIR / "lgb_model_signal_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"配置已保存: {config_path}")

    return str(model_path)


if __name__ == "__main__":
    train()
