"""信号质量预测器 —— 加载 signal_trainer 训练的模型，对买点信号打分。

与 QlibPredictor 的区别：特征从信号+日线提取（20 维），而非 45 维全量价因子。
"""
import pandas as pd, numpy as np, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.constants import DATA_DIR, BASE_DIR

SIGNAL_DIR = BASE_DIR / "data" / "signals"
MODEL_DIR = BASE_DIR / "qlib_ml" / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "lgb_model_signal.txt"
DEFAULT_CONFIG_PATH = MODEL_DIR / "lgb_model_signal_config.json"

# Regime 缓存
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
    si_path = BASE_DIR / "tmp_out" / "synthetic_index.parquet"
    if not si_path.exists():
        _regime_map = {}
        return _regime_map
    si = pd.read_parquet(si_path)
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


def _extract_signal_features(daily, signal_idx, regime_map):
    """与 signal_trainer.py 中 _extract_features 完全一致。"""
    i = signal_idx
    c = daily["close"].values.astype(float)
    h = daily["high"].values.astype(float)
    l = daily["low"].values.astype(float)
    v = daily["volume"].values.astype(float)
    o = daily["open"].values.astype(float)
    dates = daily["date"].values

    feats = []
    for lag in [5, 10, 20]:
        feats.append(c[i] / c[i - lag] - 1 if i >= lag else 0)
    for w in [5, 20, 60]:
        ma = np.mean(c[max(0, i - w + 1):i + 1])
        feats.append(c[i] / ma - 1 if ma > 0 else 0)
    for w in [10, 20]:
        feats.append(np.std(c[max(0, i - w + 1):i + 1]) / c[i] if c[i] > 0 else 0)
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    atr14 = np.mean(tr[max(0, i - 13):i + 1])
    feats.append(atr14 / c[i] if c[i] > 0 else 0)
    h60 = np.max(h[max(0, i - 59):i + 1])
    l60 = np.min(l[max(0, i - 59):i + 1])
    feats.append(c[i] / h60 - 1 if h60 > 0 else 0)
    feats.append((c[i] - l60) / (h60 - l60 + 1e-9))
    v20 = np.mean(v[max(0, i - 19):i + 1])
    feats.append(v[i] / (v20 + 1e-9) - 1)
    v5m = np.mean(v[max(0, i - 4):i + 1])
    v20m = np.mean(v[max(0, i - 19):i + 1])
    feats.append(v5m / (v20m + 1e-9) - 1)
    w = min(i + 1, 10)
    pv = np.corrcoef(c[i - w + 1:i + 1], np.log(v[i - w + 1:i + 1] + 1))[0, 1]
    feats.append(pv if not np.isnan(pv) else 0)
    feats.append(1.0 if c[i] > o[i] and v[i] > v20 else 0.0)
    for w in [10, 20]:
        diffs = np.diff(c[max(0, i - w):i + 1])
        feats.append(np.sum(diffs > 0) / max(len(diffs), 1))
    sig_date = pd.Timestamp(dates[i])
    regime = regime_map.get(sig_date, "CHOP")
    feats.append(1.0 if regime == "BULL" else 0.0)
    feats.append(1.0 if regime == "BEAR" else 0.0)
    feats.append(1.0 if regime == "CHOP" else 0.0)
    while len(feats) < 20:
        feats.append(0.0)
    return [float(v) if not np.isnan(v) else 0.0 for v in feats[:20]]


class SignalQlibPredictor:
    """信号质量预测器 —— 兼容 L4Ranker 的 qlib_predictor 接口。"""

    def __init__(self, model_path=None):
        import lightgbm as lgb
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self._model = lgb.Booster(model_file=str(self.model_path))
        if DEFAULT_CONFIG_PATH.exists():
            self.config = json.loads(DEFAULT_CONFIG_PATH.read_text())
        else:
            self.config = {}

    def score(self, codes, signal_date=None):
        """对一组股票的当前买点信号打分。

        对每只股票：读日线 → 找最近的实际买点信号 → 提取特征 → 预测 prob_good。
        返回 pd.Series (code → prob rank percentile)。

        如果某只股票没有有效信号或数据不足，跳过。
        """
        if not codes:
            return pd.Series(dtype=float)
        regime_map = _load_regime()
        results = {}
        buy_col = "日线_D1B_BUY1"

        for code in codes:
            try:
                # 读日线
                daily = pd.read_parquet(Path(DATA_DIR) / f"{code}.parquet")
                daily = daily.sort_values("date").reset_index(drop=True)
                daily["date"] = pd.to_datetime(daily["date"])

                # 读信号文件
                sf = Path(SIGNAL_DIR) / f"{code}.parquet"
                if not sf.exists():
                    continue
                sdf = pd.read_parquet(sf)
                sdf["dt"] = pd.to_datetime(sdf["dt"])

                # 找最近的有效买点
                sdf["_bt"] = sdf[buy_col].apply(lambda x: next((t for t in ["一买","二买","三买"] if t in str(x)), None))
                sdf = sdf[sdf["_bt"].notna()].sort_values("dt", ascending=False)
                if sdf.empty:
                    continue

                latest = sdf.iloc[0]
                sig_date = pd.Timestamp(latest["dt"])
                if signal_date is not None:
                    sig_date = pd.Timestamp(signal_date)

                # 在日线中定位
                mask = daily["date"] == sig_date
                if not mask.any():
                    continue
                sig_idx = daily.index[mask][0]
                if sig_idx < 120:
                    continue

                feats = _extract_signal_features(daily, sig_idx, regime_map)
                prob = float(self._model.predict(np.array(feats, dtype=np.float32).reshape(1, -1))[0])
                results[code] = prob
            except Exception:
                continue

        if not results:
            return pd.Series(dtype=float)
        scores = pd.Series(results)
        return scores.rank(pct=True) if len(scores) > 1 else pd.Series(0.5, index=scores.index)
