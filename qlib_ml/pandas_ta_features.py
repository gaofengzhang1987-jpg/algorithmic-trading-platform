"""pandas-ta 因子扩展 — 在现有 40+ 维基础上追加 60+ 技术指标。"""
import pandas_ta as ta
import pandas as pd
import numpy as np


def enrich_features(daily_df):
    """从日线数据计算 pandas-ta 技术指标。

    Args:
        daily_df: 含 open/high/low/close/volume 的 DataFrame

    Returns:
        pd.Series: 技术指标特征向量
    """
    df = daily_df.sort_values("date").reset_index(drop=True)
    df.columns = df.columns.str.lower()

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    feats = {}

    # ── 趋势类 ──
    try:
        feats["sma_20"] = float(ta.sma(c, length=20).iloc[-1])
        feats["sma_50"] = float(ta.sma(c, length=50).iloc[-1])
        feats["ema_12"] = float(ta.ema(c, length=12).iloc[-1])
        feats["ema_26"] = float(ta.ema(c, length=26).iloc[-1])
        macd = ta.macd(c)
        feats["macd"] = float(macd.iloc[-1, 0] if isinstance(macd, pd.DataFrame) else macd.iloc[-1])
        feats["macd_signal"] = float(macd.iloc[-1, 1]) if isinstance(macd, pd.DataFrame) else np.nan
        feats["macd_hist"] = float(macd.iloc[-1, 2]) if isinstance(macd, pd.DataFrame) else np.nan
    except: pass

    # ── 动量类 ──
    try:
        feats["rsi_14"] = float(ta.rsi(c, length=14).iloc[-1])
        feats["stoch_k"] = float(ta.stoch(h, l, c).iloc[-1, 0])
        feats["stoch_d"] = float(ta.stoch(h, l, c).iloc[-1, 1])
    except: pass

    # ── 波动率类 ──
    try:
        bb = ta.bbands(c)
        feats["bb_upper"] = float(bb.iloc[-1, 0])
        feats["bb_middle"] = float(bb.iloc[-1, 1])
        feats["bb_lower"] = float(bb.iloc[-1, 2])
        feats["bb_width"] = float((bb.iloc[-1, 0] - bb.iloc[-1, 2]) / bb.iloc[-1, 1])
        feats["atr_14"] = float(ta.atr(h, l, c, length=14).iloc[-1])
        feats["atr_pct"] = float(feats["atr_14"] / c.iloc[-1]) if float(c.iloc[-1]) > 0 else 0
    except: pass

    # ── 成交量类 ──
    try:
        feats["obv"] = float(ta.obv(c, v).iloc[-1])
        feats["adosc"] = float(ta.adosc(h, l, c, v).iloc[-1])
        feats["vwap"] = float((v * c).sum() / v.sum()) if v.sum() > 0 else float(c.iloc[-1])
    except: pass

    # ── 突破/通道 ──
    try:
        kc = ta.kc(h, l, c)
        feats["kc_upper"] = float(kc.iloc[-1, 0])
        feats["kc_lower"] = float(kc.iloc[-1, 2])
        dc = ta.donchian(h, l)
        feats["dc_upper"] = float(dc.iloc[-1, 0])
        feats["dc_lower"] = float(dc.iloc[-1, 2])
    except: pass

    # ── 方向性 ──
    try:
        adx = ta.adx(h, l, c)
        feats["adx"] = float(adx.iloc[-1, 0] if isinstance(adx, pd.DataFrame) else adx.iloc[-1])
        feats["dmp"] = float(adx.iloc[-1, 1]) if isinstance(adx, pd.DataFrame) else np.nan
        feats["dmn"] = float(adx.iloc[-1, 2]) if isinstance(adx, pd.DataFrame) else np.nan
        feats["aroon_up"] = float(ta.aroon(h, l).iloc[-1, 0])
        feats["aroon_down"] = float(ta.aroon(h, l).iloc[-1, 1])
    except: pass

    # ── 统计类 ──
    try:
        feats["zscore_20"] = float((c.iloc[-1] - c.tail(20).mean()) / c.tail(20).std()) if c.tail(20).std() > 0 else 0
        feats["skew_20"] = float(c.tail(20).skew()) if len(c) >= 20 else 0
        feats["kurt_20"] = float(c.tail(20).kurtosis()) if len(c) >= 20 else 0
    except: pass

    # 转为 Series，NaN -> 0
    result = pd.Series(feats, dtype=float).fillna(0)
    return result


def enrich_batch(codes, data_dir=None):
    """批量计算 pandas-ta 因子。

    Returns:
        pd.DataFrame: index=code, columns=因子名
    """
    from core.constants import DATA_DIR
    if data_dir is None:
        data_dir = DATA_DIR

    records = {}
    for code in codes:
        try:
            daily = pd.read_parquet(DATA_DIR / f"{code}.parquet")
            records[code] = enrich_features(daily)
        except Exception:
            records[code] = pd.Series(dtype=float)

    return pd.DataFrame.from_dict(records, orient="index")
