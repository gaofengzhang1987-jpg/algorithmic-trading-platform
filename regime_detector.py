"""Regime Detector — 五维牛熊评分制，精准判断 A 股市场大趋势。

纯 pandas/numpy 计算，无 CZSC 依赖。取代旧有的 2-MA 二值判断。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
logger = logging.getLogger(__name__)  # 市场状态检测


# ═══════════════════════════════════════════════════════════════
# 公共接口
# ═══════════════════════════════════════════════════════════════

def detect(data_dir=None, state_file=None):
    """返回 (regime: str, bull_score: float, dim_scores: dict)。

    regime ∈ {"BULL", "BEAR", "CHOP"}
    bull_score ∈ [0, 100]
    dim_scores: 五维得分明细 + 调试信息
    """
    base = Path(data_dir or Path(__file__).parent / "data" / "index")
    idx = pd.read_parquet(base / "000001.parquet").sort_values("date")
    idx2 = pd.read_parquet(base / "000852.parquet").sort_values("date")

    if len(idx) < 120:
        return ("CHOP", 50.0, {"error": "上证综指数据不足 120 个交易日"})

    idx = _compute_mas(idx)
    idx2 = _compute_mas(idx2)
    adx, plus_di, minus_di = _compute_adx(idx)

    last = idx.iloc[-1]
    last2 = idx2.iloc[-1]

    d1 = _score_ma_alignment(last)
    d2 = _score_price_position(last)
    d3 = _score_adx(adx[-1], plus_di[-1], minus_di[-1])
    d4 = _score_volume_price(idx)
    d5 = _score_breadth(last, last2)

    bull_score = d1 + d2 + d3 + d4 + d5

    state_path = Path(state_file or Path(__file__).parent / "tmp_out" / "regime_state.json")
    regime = _apply_anti_whipsaw(bull_score, idx, idx2, state_path)

    dim_scores = {
        "均线排列": d1,
        "价格位置": d2,
        "ADX趋势强度": round(d3, 1),
        "量价关系": d4,
        "指数协同": d5,
        "bull_score": round(bull_score, 1),
    }

    return (regime, round(bull_score, 1), dim_scores)


# ═══════════════════════════════════════════════════════════════
# 均线 & ADX 计算
# ═══════════════════════════════════════════════════════════════

def _compute_mas(df):
    for p in [5, 10, 20, 60, 120, 250]:
        df[f"MA{p}"] = df["close"].rolling(p).mean()
    return df


def _compute_adx(df, period=14):
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(close)

    tr = np.maximum(high[1:] - low[1:],
                    np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(period).mean().values

    up_move = np.diff(high)
    down_move = -np.diff(low)
    up_move[up_move < 0] = 0
    down_move[down_move < 0] = 0

    atr_safe = np.where(atr > 0, atr, 1e-10)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = pd.Series(np.concatenate([[np.nan], plus_dm])).rolling(period).mean().values / atr_safe * 100
    minus_di = pd.Series(np.concatenate([[np.nan], minus_dm])).rolling(period).mean().values / atr_safe * 100

    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).rolling(period).mean().values

    return adx, plus_di, minus_di


# ═══════════════════════════════════════════════════════════════
# 维度一：均线排列（25 分）
# ═══════════════════════════════════════════════════════════════

def _score_ma_alignment(last):
    mas = [last.get(f"MA{p}", None) for p in [5, 10, 20, 60, 120]]
    if any(m is None or (isinstance(m, float) and np.isnan(m)) for m in mas):
        return 13

    bull_pairs = sum(1 for i in range(4) if mas[i] > mas[i + 1])
    bear_pairs = sum(1 for i in range(4) if mas[i] < mas[i + 1])

    if bull_pairs == 4:
        return 25
    if bull_pairs == 3:
        return 19
    if bear_pairs == 4:
        return 0
    if bear_pairs == 3:
        return 6

    if mas[2] > 0 and abs(mas[2] - mas[3]) / mas[3] < 0.02:
        return 10
    return 13


# ═══════════════════════════════════════════════════════════════
# 维度二：价格位置（25 分）
# ═══════════════════════════════════════════════════════════════

def _score_price_position(last):
    c = last["close"]
    ma20 = last.get("MA20", 0)
    ma60 = last.get("MA60", 0)
    ma120 = last.get("MA120", 0)
    ma250 = last.get("MA250", 0)

    if isinstance(ma250, float) and np.isnan(ma250):
        ma250 = ma120 * 1.02

    if c > ma120 and ma120 > ma250:
        return 25
    elif c > ma120:
        return 20
    elif c > ma60:
        return 15
    elif c > ma20 and ma20 > 0:
        return 10
    elif c > ma60:
        return 5
    else:
        return 0


# ═══════════════════════════════════════════════════════════════
# 维度三：ADX 趋势强度（20 分）
# ═══════════════════════════════════════════════════════════════

def _score_adx(adx, plus_di, minus_di):
    if np.isnan(adx):
        return 10
    if adx >= 30 and plus_di > minus_di:
        return 20
    elif adx >= 20 and plus_di > minus_di:
        return 16
    elif adx < 20:
        return 10
    elif adx >= 20 and minus_di > plus_di:
        return 4
    elif adx >= 30 and minus_di > plus_di:
        return 0
    return 10


# ═══════════════════════════════════════════════════════════════
# 维度四：量价关系（15 分）
# ═══════════════════════════════════════════════════════════════

def _score_volume_price(df):
    vol = df["volume"].values
    close = df["close"].values

    if len(vol) < 21 or len(close) < 6:
        return 8

    vol_5 = np.mean(vol[-5:])
    vol_20 = np.mean(vol[-21:-1])
    pct_5 = (close[-1] / close[-6] - 1) * 100 if close[-6] > 0 else 0

    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0

    if vol_ratio > 1.05 and pct_5 > 0:
        return 15
    elif vol_ratio <= 1.05 and pct_5 < 0:
        return 12
    elif vol_ratio > 1.05 and pct_5 < 0:
        return 0
    elif vol_ratio <= 1.05 and pct_5 > 0:
        return 5
    return 8


# ═══════════════════════════════════════════════════════════════
# 维度五：指数协同（15 分）
# ═══════════════════════════════════════════════════════════════

def _score_breadth(last_000001, last_000852):
    s1 = last_000001["close"] > last_000001.get("MA60", 0)

    ma60_2 = last_000852.get("MA60", 0)
    if isinstance(ma60_2, float) and np.isnan(ma60_2):
        return 8
    s2 = last_000852["close"] > ma60_2

    if s1 and s2:
        return 15
    elif s1 or s2:
        return 8
    return 0


# ═══════════════════════════════════════════════════════════════
# 防抖动：3 日确认 + 状态持久化
# ═══════════════════════════════════════════════════════════════

def _compute_bull_score_for_row(idx, idx2, i):
    row = idx.iloc[i]
    idx2_row = idx2.iloc[min(i, len(idx2) - 1)]

    adx, pdi, mdi = _compute_adx(idx.iloc[:i + 1])

    d1 = _score_ma_alignment(row)
    d2 = _score_price_position(row)
    d3 = _score_adx(adx[-1], pdi[-1], mdi[-1])
    d4 = _score_volume_price(idx.iloc[:i + 1])
    d5 = _score_breadth(row, idx2_row)

    return d1 + d2 + d3 + d4 + d5


def _classify(score):
    if score >= 70:
        return "BULL"
    elif score < 30:
        return "BEAR"
    return "CHOP"


def _apply_anti_whipsaw(today_score, idx, idx2, state_path):
    new_regime = _classify(today_score)

    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"regime": "CHOP", "score_history": []}

    # 首次运行：回溯计算前 2 日的 bull_score
    if len(state["score_history"]) == 0 and len(idx) >= 63:
        for offset in [2, 1]:
            i = len(idx) - 1 - offset
            if i >= 60:
                s = _compute_bull_score_for_row(idx, idx2, i)
                state["score_history"].append({"score": round(s, 1), "regime": _classify(s)})

    state["score_history"].append({"score": round(today_score, 1), "regime": new_regime})
    state["score_history"] = state["score_history"][-5:]

    recent = state["score_history"][-3:]
    if len(recent) >= 3:
        regimes = [r["regime"] for r in recent]
        if len(set(regimes)) == 1:
            state["regime"] = regimes[0]
    else:
        state["regime"] = new_regime

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    return state["regime"]


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    regime, score, dims = detect()
    print(f"Regime: {regime}  |  Bull Score: {score}/100")
    print(f"  均线排列: {dims['均线排列']}/25")
    print(f"  价格位置: {dims['价格位置']}/25")
    print(f"  ADX趋势:  {dims['ADX趋势强度']}/20")
    print(f"  量价关系: {dims['量价关系']}/15")
    print(f"  指数协同: {dims['指数协同']}/15")
