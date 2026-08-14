"""Regime Detector — 六维牛熊评分制，精准判断 A 股市场大趋势。

维度：均线排列(20) + 价格位置(20) + ADX趋势(20) + 量价关系(15) + 指数协同(5) + 市场宽度(20) = 100 分。
纯 pandas/numpy 计算，无 CZSC 依赖。取代旧有的 2-MA 二值判断。
"""

import json
from collections import Counter
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import logging
logger = logging.getLogger(__name__)  # 市场状态检测


# ═══════════════════════════════════════════════════════════════
# 公共接口
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
DEFAULT_DAILY = BASE_DIR / "data" / "daily"
DEFAULT_INDEX = BASE_DIR / "data" / "index"
BREADTH_CACHE = BASE_DIR / "tmp_out" / "breadth_series.parquet"
MARKET_AMOUNT_CACHE = BASE_DIR / "tmp_out" / "market_amount.parquet"


def detect(data_dir=None, daily_dir=None, state_file=None,
           force_recompute_breadth=False, force_recompute_market_amount=False):
    """返回 (regime: str, bull_score: float, dim_scores: dict, breadth_pct: float)。

    regime ∈ {"BULL", "BEAR", "CHOP"}
    bull_score ∈ [0, 100]
    dim_scores: 六维得分明细 + 调试信息
    breadth_pct: 全市场 MA250 之上比例 (0-100)，仅最新日期
    """
    base = Path(data_dir or DEFAULT_INDEX)
    idx = pd.read_parquet(base / "000001.parquet").sort_values("date")
    idx2 = pd.read_parquet(base / "000852.parquet").sort_values("date")

    if len(idx) < 120:
        return ("CHOP", 50.0, {"error": "上证综指数据不足 120 个交易日"}, 50.0)

    idx = _compute_mas(idx)
    idx2 = _compute_mas(idx2)
    adx, plus_di, minus_di = _compute_adx(idx)

    last = idx.iloc[-1]
    last2 = idx2.iloc[-1]

    # 读取 000001 日线数据（_apply_anti_whipsaw 参数兼容）
    daily_dir = Path(daily_dir or DEFAULT_DAILY)
    daily_000001 = _read_daily_000001(daily_dir)

    # 市场宽度：全市场 close > MA250 比例
    latest_breadth = _get_latest_breadth(daily_dir, BREADTH_CACHE, force_recompute_breadth)
    market_amount = _get_market_amount(
        MARKET_AMOUNT_CACHE, daily_dir, force_recompute_market_amount)

    d1 = _score_ma_alignment(last)
    d2 = _score_price_position(last)
    d3 = _score_adx(adx[-1], plus_di[-1], minus_di[-1])
    d4 = _score_volume_price(idx, market_amount)
    d5 = _score_index_synergy(last, last2)
    d6 = _score_market_breadth(latest_breadth)

    bull_score = d1 + d2 + d3 + d4 + d5 + d6

    state_path = Path(state_file or BASE_DIR / "tmp_out" / "regime_state.json")
    regime = _apply_anti_whipsaw(bull_score, idx, idx2, state_path, adx[-1], latest_breadth, daily_000001)

    dim_scores = {
        "均线排列": d1,
        "价格位置": d2,
        "ADX趋势强度": round(d3, 1),
        "量价关系": d4,
        "指数协同": d5,
        "市场宽度": d6,
        "市场宽度_pct": round(latest_breadth, 1),
        "bull_score": round(bull_score, 1),
    }

    return (regime, round(bull_score, 1), dim_scores, round(latest_breadth, 1))

def _read_daily_000001(daily_dir):
    """读取 000001 日线数据（含 amount 列），用于绝对成交额参考。"""
    p = Path(daily_dir) / "000001.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).sort_values("date")


def _compute_market_amount_series(daily_dir, cache_path):
    """线上拉取沪市+深市指数成交额，求和写入全市场总成交额缓存（单位元）。

    数据源：腾讯日 K 接口，行情行第 9 个字段为成交额（万元）。
    沪市用 sh000001，深市用 sz399106（深证综指，覆盖深市全样本）。
    """
    symbols = ["sh000001", "sz399106"]
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    frames = []
    current_year = pd.Timestamp.now().year
    for symbol in symbols:
        rows_all = []
        for year in range(2019, current_year + 1):
            params = {
                "_var": "kline_dayqfq",
                "param": f"{symbol},day,{year}-01-01,{year + 1}-12-31,640,qfq",
            }
            for attempt in range(3):
                try:
                    res = requests.get(url, params=params, timeout=20)
                    text = res.text
                    payload = json.loads(text[text.find("={") + 1:])
                    rows = (payload["data"][symbol].get("day")
                            or payload["data"][symbol].get("qfqday"))
                    rows_all.extend(rows)
                    break
                except Exception:
                    if attempt == 2:
                        logger.warning("腾讯成交额拉取失败: %s %s", symbol, year)
        for row in rows_all:
            if len(row) < 9 or not row[0] or row[8] is None:
                continue
            try:
                d = pd.Timestamp(row[0]).normalize()
                amount_yuan = float(row[8]) * 1e4  # 万元 -> 元
                frames.append({"date": d, "symbol": symbol, "amount": amount_yuan})
            except Exception:
                continue
    df = pd.DataFrame(frames)
    if df.empty:
        raise RuntimeError("腾讯成交额拉取失败，无有效数据")
    # 按年请求会跨年重复返回同一日期，先按 symbol+date 去重再求和
    df = df.drop_duplicates(subset=["date", "symbol"])
    out = df.groupby("date", as_index=False)["amount"].sum()
    out.columns = ["date", "total_amount"]
    out = out.sort_values("date").reset_index(drop=True)
    out.to_parquet(cache_path, index=False)
    logger.info("全市场成交额缓存已重建: %s (%d 天)", cache_path, len(out))


def _get_market_amount(cache_path, daily_dir=None, force_recompute=False):
    """读取全市场日度总成交额缓存；缺失或落后于日线数据时自动重建。"""
    cache_path = Path(cache_path)
    need = force_recompute or not cache_path.exists()
    if not need and daily_dir is not None:
        try:
            daily_last = pd.to_datetime(
                pd.read_parquet(Path(daily_dir) / "000001.parquet",
                                columns=["date"])["date"]).max()
            cache_last = pd.to_datetime(
                pd.read_parquet(cache_path, columns=["date"])["date"]).max()
            need = cache_last < daily_last
        except Exception:
            need = True
    if need:
        _compute_market_amount_series(daily_dir or DEFAULT_DAILY, cache_path)
    try:
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["total_amount"].astype(float)
    except Exception:
        return None


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
# 维度一：均线排列（20 分）
# ═══════════════════════════════════════════════════════════════

def _score_ma_alignment(last):
    mas = [last.get(f"MA{p}", None) for p in [5, 10, 20, 60, 120]]
    if any(m is None or (isinstance(m, float) and np.isnan(m)) for m in mas):
        return 13

    bull_pairs = sum(1 for i in range(4) if mas[i] > mas[i + 1])
    bear_pairs = sum(1 for i in range(4) if mas[i] < mas[i + 1])

    if bull_pairs == 4:
        return 20
    if bull_pairs == 3:
        return 15
    if bear_pairs == 4:
        return 0
    if bear_pairs == 3:
        return 5

    if mas[2] > 0 and abs(mas[2] - mas[3]) / mas[3] < 0.02:
        return 10
    return 13


# ═══════════════════════════════════════════════════════════════
# 维度二：价格位置（20 分）
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
        return 20
    elif c > ma120:
        return 16
    elif c > ma60:
        return 12
    elif c > ma20 and ma20 > 0:
        return 8
    elif c > ma60:
        return 4
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

def _score_volume_price(df, market_amount=None):
    """量价关系评分：相对量比 + 5 日涨跌 + 全市场绝对成交额参考。

    market_amount: 全市场日度总成交额 Series（date 索引，单位元），用于流动性判断。
    阈值按两市日成交额标定：充裕 > 1.5 万亿，极度萎缩 < 6000 亿。
    """
    vol = df["volume"].values
    close = df["close"].values

    if len(vol) < 21 or len(close) < 6:
        return 8

    vol_5 = np.mean(vol[-5:])
    vol_20 = np.mean(vol[-21:-1])
    pct_5 = (close[-1] / close[-6] - 1) * 100 if close[-6] > 0 else 0

    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0

    if vol_ratio > 1.05 and pct_5 > 0:
        base = 15
    elif vol_ratio <= 1.05 and pct_5 < 0:
        base = 12
    elif vol_ratio > 1.05 and pct_5 < 0:
        base = 0
    elif vol_ratio <= 1.05 and pct_5 > 0:
        base = 5
    else:
        base = 8

    # ── 全市场绝对成交额参考（元） ──
    # 两市日成交额：牛市常见 1-3 万亿，熊市常见 5000-8000 亿
    # 阈值：近 5 日均量 > 1.5 万亿为流动性充裕，< 6000 亿为极度萎缩
    if market_amount is not None and len(market_amount) >= 21:
        dates = pd.to_datetime(df["date"])
        amounts = market_amount.reindex(dates).astype(float)
        valid = amounts.dropna()
        # 最后一天无全市场成交额（数据未更新）时跳过，避免尾部缺失误判
        if len(valid) >= 21 and valid.index[-1] == dates.iloc[-1]:
            tail = valid.tail(21)
            avg_5_amount = float(tail.tail(5).mean())
            avg_20_amount = float(tail.iloc[:-1].mean())
            # 全市场成交额 < 500 亿视为日线数据未完整更新，跳过绝对额调整
            if avg_20_amount > 0 and avg_5_amount >= 5e10:
                amount_ratio = avg_5_amount / avg_20_amount
                # 放量上涨 + 流动性充裕 → +2 信心加分
                if vol_ratio > 1.05 and pct_5 > 0 and avg_5_amount > 1.5e12 and amount_ratio > 1.0:
                    base = min(15, base + 2)
                # 成交额极度萎缩（无论量比如何）→ -2
                elif avg_5_amount < 6e11:
                    base = max(0, base - 2)

    return base


# ═══════════════════════════════════════════════════════════════
# 维度五：指数协同（5 分）
# ═══════════════════════════════════════════════════════════════

def _score_index_synergy(last_000001, last_000852):
    s1 = last_000001["close"] > last_000001.get("MA60", 0)

    ma60_2 = last_000852.get("MA60", 0)
    if isinstance(ma60_2, float) and np.isnan(ma60_2):
        return 3
    s2 = last_000852["close"] > ma60_2

    if s1 and s2:
        return 5
    elif s1 or s2:
        return 3
    return 0


# ═══════════════════════════════════════════════════════════════
# 维度六：市场宽度 — 全市场 close > MA250 比例（20 分）
# ═══════════════════════════════════════════════════════════════

def _compute_breadth_series(daily_dir, cache_path):
    """计算全市场每日 close > MA250 的股票比例，缓存到 cache_path。

    首次运行遍历所有 daily/*.parquet，后续增量更新仅追加新日期。
    返回 DataFrame: date, above_ma250, total, pct
    """
    daily_dir = Path(daily_dir)
    cache_path = Path(cache_path)

    existing = None
    latest_cached_date = None
    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        existing["date"] = pd.to_datetime(existing["date"])
        latest_cached_date = existing["date"].max()

    # 检查是否需要重新计算：抽查一个文件的最新日期
    if existing is not None and latest_cached_date is not None:
        sample_files = list(daily_dir.glob("*.parquet"))
        if sample_files:
            sample = pd.read_parquet(sample_files[0])
            sample["date"] = pd.to_datetime(sample["date"])
            if sample["date"].max() <= latest_cached_date:
                return existing

    date_counts = Counter()   # date → stocks above MA250
    date_totals = Counter()   # date → total stocks with valid MA250

    files = sorted(daily_dir.glob("*.parquet"))
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
            if df["date"].dtype != "datetime64[ns]":
                df["date"] = pd.to_datetime(df["date"])
            if len(df) < 250:
                continue
            # 增量：只计算缓存中不存在的日期
            if existing is not None and latest_cached_date is not None:
                df = df[df["date"] > latest_cached_date]
                if len(df) < 250:
                    continue
            df["MA250"] = df["close"].rolling(250).mean()
            df = df.dropna(subset=["MA250"])
            above = df[df["close"] > df["MA250"]]
            for d in above["date"]:
                date_counts[d] += 1
            for d in df["date"]:
                date_totals[d] += 1
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            logger.info("breadth progress: %d/%d", i + 1, len(files))

    if not date_totals:
        if existing is not None:
            return existing
        return pd.DataFrame(columns=["date", "above_ma250", "total", "pct"])

    dates = sorted(set(date_counts.keys()) | set(date_totals.keys()))
    records = []
    for d in dates:
        total = date_totals.get(d, 0)
        above = date_counts.get(d, 0)
        records.append({
            "date": d,
            "above_ma250": above,
            "total": total,
            "pct": round(above / total * 100, 1) if total > 0 else 0.0,
        })

    new_df = pd.DataFrame(records).sort_values("date")

    if existing is not None:
        result = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    else:
        result = new_df

    result = result.sort_values("date").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)

    return result


def _get_latest_breadth(daily_dir, cache_path, force_recompute=False):
    """获取最新日期的市场宽度百分比。首次或强制时触发全量计算并缓存。"""
    if force_recompute:
        cache_path.unlink(missing_ok=True)

    if not cache_path.exists():
        _compute_breadth_series(daily_dir, cache_path)

    df = pd.read_parquet(cache_path)
    if df.empty:
        return 50.0
    df["date"] = pd.to_datetime(df["date"])
    latest = df.iloc[-1]
    return float(latest.get("pct", 50.0))


def _score_market_breadth(breadth_pct):
    """市场宽度评分（0-20）：全市场站上 MA250 的比例。

    ≥70% → 20（牛市宽度），50-70% → 15，30-50% → 10，15-30% → 5，<15% → 0"""
    if breadth_pct >= 70:
        return 20
    elif breadth_pct >= 50:
        return 15
    elif breadth_pct >= 30:
        return 10
    elif breadth_pct >= 15:
        return 5
    else:
        return 0


# ═══════════════════════════════════════════════════════════════
# 防抖动：3 日确认 + 状态持久化
# ═══════════════════════════════════════════════════════════════

def _compute_bull_score_for_row(idx, idx2, i):
    """回溯计算第 i 日的 bull_score（五维，不含市场宽度，用于防抖动历史回填）。"""
    row = idx.iloc[i]
    idx2_row = idx2.iloc[min(i, len(idx2) - 1)]

    adx, pdi, mdi = _compute_adx(idx.iloc[:i + 1])

    d1 = _score_ma_alignment(row)
    d2 = _score_price_position(row)
    d3 = _score_adx(adx[-1], pdi[-1], mdi[-1])
    d4 = _score_volume_price(idx.iloc[:i + 1])
    d5 = _score_index_synergy(row, idx2_row)

    return d1 + d2 + d3 + d4 + d5


def _classify(score, adx_val=None, breadth_pct=None):
    """综合评分 → Regime 分类。震荡市有正向判定通道。

    正向 CHOP 判定：ADX < 20（无趋势）且市场宽度 30-70%（无极端分化）→ 强震荡信号，
    即使复合评分偏向牛/熊也以 CHOP 归类，反映"无趋势+结构分化"的真实市场状态。
    """
    # CHOP 正向判定：ADX 低 + 宽度居中 → 震荡优先
    if adx_val is not None and breadth_pct is not None:
        if adx_val < 20 and 30 <= breadth_pct <= 70:
            return "CHOP"

    if score >= 70:
        return "BULL"
    elif score < 30:
        return "BEAR"
    return "CHOP"


def _apply_anti_whipsaw(today_score, idx, idx2, state_path,
                         adx_val=None, breadth_pct=None, daily_000001=None):
    new_regime = _classify(today_score, adx_val, breadth_pct)

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
                state["score_history"].append({"score": round(s, 1), "regime": _classify(s, None, None)})

    state["score_history"].append({"score": round(today_score, 1), "regime": new_regime})
    state["score_history"] = state["score_history"][-5:]

    recent = state["score_history"][-3:]
    if len(recent) >= 3:
        regimes = [r["regime"] for r in recent]
        if len(set(regimes)) == 1:
            # CZSC 结构确认：BULL/BEAR 切换时验证中枢+背驰+分型
            old_regime = state.get("regime", "CHOP")
            if regimes[0] == "CHOP":
                # CHOP 免检，允许从 BULL/BEAR 切回震荡
                state["regime"] = "CHOP"
            elif regimes[0] != old_regime and regimes[0] in ("BULL", "BEAR"):
                confirmed, reason = _confirm_regime_change(regimes[0], daily_000001, idx)
                if not confirmed:
                    logger.warning("[%s_REJECT] %s — 保持 %s", regimes[0], reason, old_regime)
                    # 不切换，记录当前 regime 评分但不改变 regime
                    state["score_history"][-1]["regime"] = old_regime
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
                    return old_regime
                logger.info("[%s_CONFIRM] %s", regimes[0], reason)
                state["regime"] = regimes[0]
    else:
        state["regime"] = new_regime

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    return state["regime"]


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# CZSC 结构确认：Regime 切换时的缠论验证
# ═══════════════════════════════════════════════════════════════

def _detect_pivots_from_czsc(czsc):
    """从 CZSC 笔列表检测中枢，返回中枢列表 [{"dir","zg","zd","gg","dd","sdt","edt"}]。"""
    pivots = []
    bl = czsc.bi_list
    i = 0
    while i < len(bl) - 2:
        a, b, c = bl[i], bl[i + 1], bl[i + 2]
        oh = min(a.high, b.high, c.high)
        ol = max(a.low, b.low, c.low)
        if oh > ol:
            pb, zg, zd, gg_val, dd_val = [a, b, c], oh, ol, oh, ol
            j = i + 3
            while j < len(bl):
                nx = bl[j]
                nx_h = max(nx.high, nx.low)
                nx_l = min(nx.high, nx.low)
                if nx_h < zg or nx_l > zd:
                    break
                zg = max(zg, nx.high)
                zd = min(zd, nx.low)
                gg_val = max(gg_val, nx.high)
                dd_val = min(dd_val, nx.low)
                pb.append(nx)
                j += 1
            from czsc.objects import Direction
            pivots.append({"dir": "上涨" if b.direction == Direction.Up else "下跌",
                           "zg": zg, "zd": zd, "gg": gg_val, "dd": dd_val,
                           "sdt": pb[0].sdt, "edt": pb[-1].edt})
            i = j
        else:
            i += 1
    return pivots


def _bi_macd_hist_sum(dates, close_arr, bi):
    """计算一根笔对应 bar 段的 MACD 柱面积（绝对值和）。"""
    # 确保 dates 是 datetime64，兼容 pandas 和 numpy 类型
    dates = pd.to_datetime(dates)
    mask = (dates >= bi.sdt) & (dates <= bi.edt)
    if mask.sum() < 3:
        return 0.0
    seg_close = close_arr[mask]
    ema12 = pd.Series(seg_close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(seg_close).ewm(span=26, adjust=False).mean().values
    macd = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    hist = np.abs(macd - signal)
    return float(np.sum(hist))


def _check_top_divergence(czsc, close_arr, dates):
    """检查最后一个向上笔是否顶背驰。返回 (has_divergence: bool, detail: str)。"""
    from czsc.objects import Direction
    up_bis = [bi for bi in czsc.bi_list if bi.direction == Direction.Up]
    if len(up_bis) < 2:
        return (False, "向上笔不足2根")
    a, b = up_bis[-2], up_bis[-1]
    area_a = _bi_macd_hist_sum(dates, close_arr, a)
    area_b = _bi_macd_hist_sum(dates, close_arr, b)
    if area_a <= 0:
        return (False, f"前笔面积=0")
    ratio = area_b / area_a
    has_div = b.high > a.high and ratio < 0.9
    detail = f"high {a.high:.1f}→{b.high:.1f} area {area_a:.1f}→{area_b:.1f} ratio={ratio:.2f}"
    return (has_div, detail)


def _check_bottom_divergence(czsc, close_arr, dates):
    """检查最后一个向下笔是否底背驰。返回 (has_divergence: bool, detail: str)。"""
    from czsc.objects import Direction
    down_bis = [bi for bi in czsc.bi_list if bi.direction == Direction.Down]
    if len(down_bis) < 2:
        return (False, "向下笔不足2根")
    a, b = down_bis[-2], down_bis[-1]
    area_a = _bi_macd_hist_sum(dates, close_arr, a)
    area_b = _bi_macd_hist_sum(dates, close_arr, b)
    if area_a <= 0:
        return (False, f"前笔面积=0")
    ratio = area_b / area_a
    has_div = b.low < a.low and ratio < 0.9
    detail = f"low {a.low:.2f}→{b.low:.2f} area {area_a:.1f}→{area_b:.1f} ratio={ratio:.2f}"
    return (has_div, detail)


def _get_recent_fx(czsc, direction="底"):
    """获取最近的分型（底分型或顶分型）。返回 {"dt","high","low"} 或 None。"""
    if not hasattr(czsc, "fx_list") or not czsc.fx_list:
        return None
    if direction == "底":
        for fx in reversed(czsc.fx_list):
            if hasattr(fx, "mark") and "底分型" in str(fx.mark):
                return fx
    else:
        for fx in reversed(czsc.fx_list):
            if hasattr(fx, "mark") and "顶分型" in str(fx.mark):
                return fx
    return None


def _confirm_bull(czsc, pivots, close, df):
    """BULL 结构确认：上涨中枢 + 价格在 ZG 上 + 无顶背驰 + 底分型在 ZG 上。"""
    dates = df["date"].values
    close_arr = df["close"].values

    up_pivots = [p for p in pivots if p["dir"] == "上涨"]
    if not up_pivots:
        return (False, "无上涨中枢 — 均线上穿缺乏结构支撑")
    last_up = up_pivots[-1]
    zg = last_up["zg"]

    if close < zg:
        return (False, f"价格 {close:.1f} < ZG {zg:.1f} — 未站上中枢上沿")

    has_div, div_detail = _check_top_divergence(czsc, close_arr, dates)
    if has_div:
        return (False, f"顶背驰: {div_detail} — 趋势末端强弩之末")

    btm_fx = _get_recent_fx(czsc, "底")
    if btm_fx is None:
        return (False, "无底分型记录 — 无法确认三买结构")
    if btm_fx.low < zg:
        return (False, f"底分型低点 {btm_fx.low:.2f} < ZG {zg:.1f} — 回调可能回中枢，非三买")

    return (True, f"ZG={zg:.1f} close={close:.1f} 无顶背驰 底分型在ZG上方 → 结构确认")


def _confirm_bear(czsc, pivots, close, df):
    """BEAR 结构确认：下跌中枢 + 价格在 ZD 下 + 无底背驰 + 顶分型在 ZD 下。"""
    dates = df["date"].values
    close_arr = df["close"].values

    down_pivots = [p for p in pivots if p["dir"] == "下跌"]
    if not down_pivots:
        return (False, "无下跌中枢 — 恐慌性下跌缺乏结构确认")
    last_down = down_pivots[-1]
    zd = last_down["zd"]

    if close > zd:
        return (False, f"价格 {close:.1f} > ZD {zd:.1f} — 仍在震荡区间内")

    has_div, div_detail = _check_bottom_divergence(czsc, close_arr, dates)
    if has_div:
        return (False, f"底背驰: {div_detail} — 熊末信号，不宜确认 BEAR")

    top_fx = _get_recent_fx(czsc, "顶")
    if top_fx is None:
        return (False, "无顶分型记录 — 无法确认三卖结构")
    if top_fx.high > zd:
        return (False, f"顶分型高点 {top_fx.high:.2f} > ZD {zd:.1f} — 反弹可能回中枢，非三卖")

    return (True, f"ZD={zd:.1f} close={close:.1f} 无底背驰 顶分型在ZD下方 → 结构确认")


def _confirm_regime_change(new_regime, daily_000001, idx):
    """CZSC 结构确认 regime 切换。返回 (confirmed: bool, reason: str)。"""
    if new_regime not in ("BULL", "BEAR"):
        return (True, "CHOP 免检")

    if idx is None or len(idx) < 300:
        return (True, f"日线数据不足({len(idx) if idx is not None else 0}bar)，跳过结构确认")

    try:
        from czsc import CZSC, RawBar, Freq
        df = idx.sort_values("date").reset_index(drop=True)
        bars = [RawBar(symbol="000001", id=j + 1, dt=r["date"].to_pydatetime(),
                       freq=Freq.D, open=r["open"], close=r["close"],
                       high=r["high"], low=r["low"],
                       vol=r.get("volume", 0), amount=0)
                for j, (_, r) in enumerate(df.iterrows())]
        czsc = CZSC(bars, max_bi_num=50)
    except Exception as e:
        logger.warning("CZSC 初始化失败，跳过结构确认: %s", e)
        return (True, f"CZSC 异常: {e}")

    if len(czsc.bi_list) < 4:
        return (True, f"笔数不足({len(czsc.bi_list)})，跳过结构确认")

    pivots = _detect_pivots_from_czsc(czsc)
    close = float(df["close"].iloc[-1])

    if new_regime == "BULL":
        return _confirm_bull(czsc, pivots, close, df)
    else:
        return _confirm_bear(czsc, pivots, close, df)
if __name__ == "__main__":
    regime, score, dims, breadth_pct = detect()
    print(f"Regime: {regime}  |  Bull Score: {score}/100  |  市场宽度: {breadth_pct}%")
    print(f"  均线排列: {dims['均线排列']}/20")
    print(f"  价格位置: {dims['价格位置']}/20")
    print(f"  ADX趋势:  {dims['ADX趋势强度']}/20")
    print(f"  量价关系: {dims['量价关系']}/15")
    print(f"  指数协同: {dims['指数协同']}/5")
    print(f"  市场宽度: {dims['市场宽度']}/20")
