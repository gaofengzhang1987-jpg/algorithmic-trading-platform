"""LightGBM 因子训练管道 — 直接从 parquet 日线数据提取因子并训练。

不依赖 Qlib 数据格式。手动计算 Alpha158 风格的 40+ 维技术因子。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from core.constants import DATA_DIR, BASE_DIR

MODEL_DIR = BASE_DIR / "qlib_ml" / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "lgb_model.txt"


def _extract_single_features(daily, horizon=20):
    from qlib_ml.features import extract_62
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    n = len(daily)
    min_samples = 120
    if n < min_samples + horizon:
        return [], [], []
    features, labels, dates_out = [], [], []
    for i in range(min_samples, n - horizon):
        window = daily.iloc[i-min_samples:i+1]
        feats_arr = extract_62(window)
        if feats_arr is None:
            continue
        fwd_price = daily["close"].iloc[i + horizon]
        features.append(feats_arr.tolist())
        labels.append(fwd_price / daily["close"].iloc[i] - 1)
        dates_out.append(daily["date"].iloc[i])
    return features, labels, dates_out


def train(
    train_period=("2020-01-01", "2024-12-31"),
    valid_period=("2025-01-01", "2025-06-30"),
    horizon=20,
    model_name="lgb_model",
    max_stocks=500,
):
    """训练 LightGBM 模型预测 forward return。

    Args:
        train_period: (start, end) 训练区间
        valid_period: (start, end) 验证区间
        horizon: 预测周期（交易日）
        model_name: 模型文件名
        max_stocks: 最大训练股票数
    """
    import lightgbm as lgb

    codes = sorted(p.stem for p in Path(DATA_DIR).glob("*.parquet") if p.stem.isdigit())
    if max_stocks:
        codes = codes[:max_stocks]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.txt"

    # ── 缓存路径 ──
    CACHE_DIR = MODEL_DIR / "cache"
    cache_X = CACHE_DIR / "cache_X.npy"
    cache_y = CACHE_DIR / "cache_y.npy"
    cache_dates = CACHE_DIR / "cache_dates.npy"
    cache_meta = CACHE_DIR / "cache_meta.json"

    # ── 数据指纹（股票数 + 最新日期）──
    _latest = ""
    for f in Path(DATA_DIR).glob("*.parquet"):
        _df = pd.read_parquet(f, columns=["date"])
        _d = str(_df["date"].max())
        if _d > _latest:
            _latest = _d
    fingerprint = (len(codes), _latest)

    # ── 检查缓存 ──
    hit = False
    if cache_meta.exists():
        try:
            _meta = json.loads(cache_meta.read_text())
            if (_meta.get("n_codes") == fingerprint[0]
                    and _meta.get("data_latest") == fingerprint[1]
                    and _meta.get("n_features") == 62):
                print(f"缓存命中 ({fingerprint[0]} 只, 最新 {fingerprint[1]}, 62 维), 从缓存加载...")
                X_arr = np.load(cache_X, mmap_mode="r")
                y_arr = np.load(cache_y, mmap_mode="r")
                all_dates = np.load(cache_dates, allow_pickle=True)
                hit = True
        except Exception:
            pass

    if not hit:
        # ── 收集训练数据 ──
        print(f"提取因子: {len(codes)} 只股票, horizon={horizon}...")
        all_X, all_y, all_dates = [], [], []
        errors = 0

        for ci, code in enumerate(codes):
            try:
                df = pd.read_parquet(Path(DATA_DIR) / f"{code}.parquet")
                feats, labels, dates = _extract_single_features(df, horizon)
                if feats:
                    all_X.extend(feats)
                    all_y.extend(labels)
                    all_dates.extend(dates)
            except Exception:
                errors += 1
            if (ci + 1) % 100 == 0:
                print(f"  进度: {ci+1}/{len(codes)}, 样本: {len(all_X)}, 错误: {errors}")

        print(f"总样本: {len(all_X)}, 错误股票: {errors}")

        if len(all_X) < 1000:
            raise RuntimeError(f"样本不足: {len(all_X)}，至少需要 1000")

        X_arr = np.array(all_X, dtype=np.float32)
        y_arr = np.array(all_y, dtype=np.float32)
        all_dates = np.array(all_dates)

        # ── 写入缓存 ──
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_X, X_arr)
        np.save(cache_y, y_arr)
        np.save(cache_dates, all_dates)
        cache_meta.write_text(json.dumps({
            "n_codes": fingerprint[0], "data_latest": fingerprint[1],
            "n_samples": len(X_arr), "n_features": X_arr.shape[1],
            "created_at": datetime.now().isoformat(),
        }, indent=2))
        print(f"因子已缓存到 {CACHE_DIR}")

    # NaN 处理
    nan_mask = np.isnan(X_arr).any(axis=1) | np.isnan(y_arr)
    X_clean = X_arr[~nan_mask]
    y_clean = y_arr[~nan_mask]
    dates_clean = np.array(all_dates)[~nan_mask]
    print(f"剔除 NaN 后: {len(X_clean)} 条 ({len(X_arr) - len(X_clean)} 条 NaN)")

    # ── 时间分割 ──
    train_start = pd.Timestamp(train_period[0])
    train_end = pd.Timestamp(train_period[1])
    valid_start = pd.Timestamp(valid_period[0])
    valid_end = pd.Timestamp(valid_period[1])

    dates_pd = pd.to_datetime(dates_clean)
    train_mask = (dates_pd >= train_start) & (dates_pd <= train_end)
    valid_mask = (dates_pd >= valid_start) & (dates_pd <= valid_end)

    X_train, y_train = X_clean[train_mask], y_clean[train_mask]
    X_valid, y_valid = X_clean[valid_mask], y_clean[valid_mask]

    print(f"训练集: {len(X_train)}, 验证集: {len(X_valid)}")

    if len(X_train) < 500:
        raise RuntimeError(f"训练集不足: {len(X_train)}")

    # ── 训练 ──
    print("训练 LightGBM...")
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 200,
        "verbose": -1,
        "num_threads": 4,
    }

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)],
    )

    model.save_model(str(model_path))
    print(f"模型已保存: {model_path}")

    # 评估
    y_pred = model.predict(X_valid)
    corr = np.corrcoef(y_valid, y_pred)[0, 1]
    rmse_val = np.sqrt(np.mean((y_valid - y_pred)**2))
    print(f"验证集: n={len(y_valid)}, corr={corr:.4f}, rmse={rmse_val:.4f}")

    # 保存配置（含评估指标）
    config_path = MODEL_DIR / f"{model_name}_config.json"
    config_path.write_text(json.dumps({
        "train_period": list(train_period),
        "valid_period": list(valid_period),
        "horizon": horizon,
        "model_path": str(model_path),
        "n_features": X_train.shape[1],
        "n_samples": len(X_clean),
        "corr": round(float(corr), 6),
        "rmse": round(float(rmse_val), 6),
        "created_at": datetime.now().isoformat(),
    }, indent=2, ensure_ascii=False))

    return str(model_path)
