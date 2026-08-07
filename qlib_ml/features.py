"""统一特征提取 — 40 手工因子 + 31 pandas-ta = ~71 维。trainer 和 predictor 共用。"""
import numpy as np
import pandas as pd


def extract_handcrafted(close, high, low, volume, n):
    """提取 40 维手工因子（与现有 trainer 一致）。

    Returns: list of float (len=40)
    """
    feats = []
    wc = close[-121:]  # 120 bar window + current
    wh = high[-121:]
    wl = low[-121:]
    wv = volume[-121:]
    i = len(wc) - 1

    # 收益类 (5)
    for lag in [1, 5, 10, 20, 60]:
        feats.append(wc[i] / wc[i - lag] - 1 if i >= lag else 0)
    # MA 偏离 (4)
    for w in [5, 10, 20, 60]:
        ma = np.mean(wc[-w:])
        feats.append(wc[-1] / ma - 1 if ma != 0 else 0)
    # 波动率 (3)
    for w in [5, 10, 20]:
        feats.append(np.std(wc[-w:]) / wc[-1])
    # 高低价位置 (4)
    feats.append(wh[-1] / wc[-1] - 1)
    feats.append(wl[-1] / wc[-1] - 1)
    d = wh[-1] - wl[-1] + 1e-12
    feats.append((wc[-1] - wl[-1]) / d)
    feats.append((wh[-1] - wl[-1]) / wc[-1])
    # 价格动量 (3)
    feats.append((wh[-1] - wh[-2]) / wc[-1])
    feats.append((wl[-1] - wl[-2]) / wc[-1])
    feats.append((wc[-1] - wc[-2]) / wc[-2])
    # N 日高/低 (8)
    for w in [5, 10, 20, 60]:
        feats.append(np.max(wh[-w:]) / wc[-1] - 1)
        feats.append(np.min(wl[-w:]) / wc[-1] - 1)
    # 成交量变化 (4)
    feats.append(wv[-1] / (wv[-2] + 1e-12) - 1)
    for w in [5, 10, 20]:
        feats.append(np.mean(wv[-w:]) / (np.mean(wv[-w*2:-w]) + 1e-12) - 1)
    # 成交量均比 (3)
    for w in [5, 10, 20]:
        feats.append(wv[-1] / (np.mean(wv[-w:]) + 1e-12) - 1)
    # 价量相关 (3)
    for w in [5, 10, 20]:
        c = np.corrcoef(wc[-w:], np.log(wv[-w:] + 1))[0, 1]
        feats.append(c if not np.isnan(c) else 0)
    # MA 交叉 (3)
    for s, l in [(5, 10), (5, 20), (10, 20)]:
        feats.append(np.mean(wc[-s:]) / (np.mean(wc[-l:]) + 1e-12) - 1)
    # RSI (1)
    diffs = np.diff(wc[-15:])
    g, l = np.maximum(diffs, 0), np.maximum(-diffs, 0)
    rs = np.mean(g) / (np.mean(l) + 1e-12)
    feats.append(100 - 100 / (1 + rs))
    # 趋势强度 (2)
    for w in [10, 20]:
        feats.append(np.sum(np.diff(wc[-w:]) > 0) / w)
    # 振幅 + 趋势速度 (2)
    feats.append((np.max(wh[-20:]) - np.min(wl[-20:])) / wc[-1])
    feats.append((wc[-1] - wc[-20]) / wc[-20] / 20)
    return feats


def extract_pandas_ta(daily_df):
    """提取 31 维 pandas-ta 技术指标。

    daily_df 需含 open/high/low/close/volume 列。
    Returns: list of float (len=31)
    """
    try:
        import pandas_ta as ta
    except ImportError:
        return [0] * 31

    df = daily_df.sort_values("date").tail(120).reset_index(drop=True)
    df.columns = df.columns.str.lower()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    def _f(x):
        try:
            return float(x.iloc[-1])
        except:
            return 0.0

    feats = []

    # 趋势 (7)
    try: feats.append(_f(ta.sma(c, length=20)))
    except: feats.append(0)
    try: feats.append(_f(ta.sma(c, length=50)))
    except: feats.append(0)
    try: feats.append(_f(ta.ema(c, length=12)))
    except: feats.append(0)
    try: feats.append(_f(ta.ema(c, length=26)))
    except: feats.append(0)
    try:
        macd = ta.macd(c)
        if isinstance(macd, pd.DataFrame):
            feats.append(_f(macd.iloc[:, 0])); feats.append(_f(macd.iloc[:, 1])); feats.append(_f(macd.iloc[:, 2]))
        else:
            feats.extend([0, 0, 0])
    except: feats.extend([0, 0, 0])

    # 动量 (3)
    try: feats.append(_f(ta.rsi(c, length=14)))
    except: feats.append(0)
    try:
        st = ta.stoch(h, l, c)
        if isinstance(st, pd.DataFrame):
            feats.append(_f(st.iloc[:, 0])); feats.append(_f(st.iloc[:, 1]))
        else:
            feats.extend([0, 0])
    except: feats.extend([0, 0])

    # 波动率 (5)
    try:
        bb = ta.bbands(c)
        feats.append(_f(bb.iloc[:, 0])); feats.append(_f(bb.iloc[:, 1])); feats.append(_f(bb.iloc[:, 2]))
        feats.append((_f(bb.iloc[:, 0]) - _f(bb.iloc[:, 2])) / (_f(bb.iloc[:, 1]) + 1e-9))
    except: feats.extend([0, 0, 0, 0])
    try: feats.append(_f(ta.atr(h, l, c, length=14)) / (_f(c) + 1e-9))
    except: feats.append(0)

    # 成交量 (3)
    try: feats.append(_f(ta.obv(c, v)))
    except: feats.append(0)
    try: feats.append(_f(ta.adosc(h, l, c, v)))
    except: feats.append(0)
    try: feats.append(float((v * c).sum() / v.sum()) if v.sum() > 0 else _f(c))
    except: feats.append(0)

    # 通道 (4)
    try:
        kc = ta.kc(h, l, c)
        feats.append(_f(kc.iloc[:, 0])); feats.append(_f(kc.iloc[:, 2]))
    except: feats.extend([0, 0])
    try:
        dc = ta.donchian(h, l)
        feats.append(_f(dc.iloc[:, 0])); feats.append(_f(dc.iloc[:, 2]))
    except: feats.extend([0, 0])

    # 方向 (5)
    try:
        adx = ta.adx(h, l, c)
        feats.append(_f(adx.iloc[:, 0])); feats.append(_f(adx.iloc[:, 1])); feats.append(_f(adx.iloc[:, 2]))
    except: feats.extend([0, 0, 0])
    try:
        ar = ta.aroon(h, l)
        feats.append(_f(ar.iloc[:, 0])); feats.append(_f(ar.iloc[:, 1]))
    except: feats.extend([0, 0])

    # 统计 (3)
    try: feats.append((_f(c) - c.tail(20).mean()) / (c.tail(20).std() + 1e-9))
    except: feats.append(0)
    try: feats.append(float(c.tail(20).skew()))
    except: feats.append(0)
    try: feats.append(float(c.tail(20).kurtosis()))
    except: feats.append(0)

    # 补齐到 31
    while len(feats) < 31:
        feats.append(0)

    return [float(v) if not np.isnan(v) else 0.0 for v in feats]


def extract_all(daily_df):
    """提取全量特征: 40(手工) + 31(pandas-ta) = 71 维。

    Args:
        daily_df: 含 open/high/low/close/volume/date 的 DataFrame

    Returns:
        np.ndarray shape (71,)
    """
    c = daily_df["close"].values.astype(float)
    h = daily_df["high"].values.astype(float)
    l = daily_df["low"].values.astype(float)
    v = daily_df["volume"].values.astype(float)
    n = len(c)

    if n < 121:
        return None

    hc = extract_handcrafted(c, h, l, v, n)
    ta = extract_pandas_ta(daily_df)

    return np.array(hc + ta, dtype=np.float32)


# ── 17 个与手工因子不相关的新因子 ──
def extract_pandas_ta_novel(daily_df):
    """提取 17 个与手工 45 维不相关的 pandas-ta 指标（|corr| < 0.50）。"""
    import pandas_ta as ta
    df = daily_df.sort_values("date").tail(120).reset_index(drop=True)
    df.columns = df.columns.str.lower()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    def _f(x):
        try: return float(x.iloc[-1])
        except: return 0.0

    feats = []
    # MA (4): sma20, sma50, ema12, ema26
    try: feats.append(_f(ta.sma(c, length=20)))
    except: feats.append(0)
    try: feats.append(_f(ta.sma(c, length=50)))
    except: feats.append(0)
    try: feats.append(_f(ta.ema(c, length=12)))
    except: feats.append(0)
    try: feats.append(_f(ta.ema(c, length=26)))
    except: feats.append(0)

    # Bollinger (3): bb_up, bb_mid, bb_low
    try:
        bb = ta.bbands(c)
        feats.append(_f(bb.iloc[:,0])); feats.append(_f(bb.iloc[:,1])); feats.append(_f(bb.iloc[:,2]))
    except: feats.extend([0,0,0])

    # Volume (3): obv, adosc, vwap
    try: feats.append(_f(ta.obv(c, v)))
    except: feats.append(0)
    try: feats.append(_f(ta.adosc(h, l, c, v)))
    except: feats.append(0)
    try: feats.append(float((v * c).sum() / v.sum()) if v.sum() > 0 else _f(c))
    except: feats.append(0)

    # Channels (4): kc_up, kc_low, dc_up, dc_low
    try:
        kc = ta.kc(h, l, c)
        feats.append(_f(kc.iloc[:,0])); feats.append(_f(kc.iloc[:,2]))
    except: feats.extend([0,0])
    try:
        dc = ta.donchian(h, l)
        feats.append(_f(dc.iloc[:,0])); feats.append(_f(dc.iloc[:,2]))
    except: feats.extend([0,0])

    # Direction (3): adx, dmp, dmn
    try:
        adx = ta.adx(h, l, c)
        feats.append(_f(adx.iloc[:,0])); feats.append(_f(adx.iloc[:,1])); feats.append(_f(adx.iloc[:,2]))
    except: feats.extend([0,0,0])

    return [float(v) if not np.isnan(v) else 0.0 for v in feats]


def extract_62(daily_df):
    """45 手工 + 17 不相关 pandas-ta = 62 维。"""
    c = daily_df["close"].values.astype(float)
    h = daily_df["high"].values.astype(float)
    l = daily_df["low"].values.astype(float)
    v = daily_df["volume"].values.astype(float)
    n = len(c)
    if n < 121:
        return None
    hc = extract_handcrafted(c, h, l, v, n)
    ta_novel = extract_pandas_ta_novel(daily_df)
    return np.array(hc + ta_novel, dtype=np.float32)
