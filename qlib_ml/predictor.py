"""LightGBM 预测评分器 — 加载直接训练的模型，对 L2 候选池打分。"""
import pandas as pd, numpy as np, json
from pathlib import Path
from core.constants import BASE_DIR, DATA_DIR
from qlib_ml.features import extract_handcrafted, extract_62

MODEL_DIR = BASE_DIR / "qlib_ml" / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "lgb_model.txt"
DEFAULT_CONFIG_PATH = MODEL_DIR / "lgb_model_config.json"


class QlibPredictor:
    """LightGBM 模型预测器。"""

    def __init__(self, model_path=None):
        import lightgbm as lgb
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model_path = Path(model_path)
        self.horizon = 20
        self.n_features = 45
        if DEFAULT_CONFIG_PATH.exists():
            cfg = json.loads(DEFAULT_CONFIG_PATH.read_text())
            self.horizon = cfg.get("horizon", 20)
            self.n_features = cfg.get("n_features", 45)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self._model = lgb.Booster(model_file=str(self.model_path))

    def score(self, codes, signal_date=None):
        if not codes:
            return pd.Series(dtype=float)
        results = {}
        for code in codes:
            try:
                df = pd.read_parquet(Path(DATA_DIR) / f"{code}.parquet")
                df = df.sort_values("date").reset_index(drop=True)
                df["date"] = pd.to_datetime(df["date"])
                if signal_date:
                    df = df[df["date"] <= pd.Timestamp(signal_date)]
                if len(df) < 121:
                    continue
                if self.n_features == 62:
                    feats_arr = extract_62(df)
                    feats = feats_arr.tolist() if feats_arr is not None else None
                else:
                    feats = extract_handcrafted(
                        df["close"].values.astype(float),
                        df["high"].values.astype(float),
                        df["low"].values.astype(float),
                        df["volume"].values.astype(float),
                        len(df))
                if feats is None: continue
                raw_pred = float(self._model.predict(np.array(feats, dtype=np.float32).reshape(1, -1))[0])
                results[code] = raw_pred
            except Exception:
                continue
        if not results:
            return pd.Series(dtype=float)
        scores = pd.Series(results)
        return scores.rank(pct=True) if len(scores) > 1 else pd.Series(0.5, index=scores.index)
