"""EntryFilter — L1→L2 入场过滤器。从 backtest.py 分离，独立维护。"""
from collections import namedtuple
import pandas as pd
from pathlib import Path
from czsc.objects import Direction
from core.constants import BASE_DIR

from zone2_pattern import _score_stroke, _score_volume, _score_macd, _is_bottom_fractal
from scoring.common import CommonScoringMixin
from scoring.buy1 import Buy1ScoringMixin
from scoring.buy2 import Buy2ScoringMixin
from scoring.buy3 import Buy3ScoringMixin

FilterResult = namedtuple("FilterResult", ["passed", "buy_type", "total_score", "dimension_scores", "reject_reason"])

REGIME_WEIGHTS = {
    "一买": {
        "BEAR": {"中枢": 0.57, "量比": 0.44, "MACD": 0.29, "底分型": 0.13, "核心验证": 0.10, "底部放量": 0.10},
        "BULL": {"量比": 0.70, "中枢": 0.45, "MACD": 0.39, "核心验证": 0.34, "距离": 0.26},
        "CHOP": {"核心验证": 0.5877719770252653, "量比": 1.9296510587242894, "MACD": 0.07211783281354563, "底分型": 2.0496247460344184, "中枢": 2.312864421353475, "距离": 1.267394016376168, "区间位置": 1.6079882498225504, "波动压缩": 0.623516358220845, "时间消化": 0.08210690435719653, "相对强度": 0.6620315182610395},
        "CHOP": {"核心验证": 0.59, "量比": 1.93, "MACD": 0.07, "底分型": 2.05, "中枢": 2.31, "距离": 1.27, "区间位置": 1.61, "波动压缩": 0.62, "时间消化": 0.08, "相对强度": 0.66},
    },
    "二买": {
        "BEAR": {"MACD": 1.52, "量比": 0.90, "底分型": 0.44, "核心验证": 0.40},
        "BULL": {"MA防守": 2.96, "MACD": 0.99, "笔数": 0.98, "底分型": 0.08},
        "CHOP": {"MACD": 2.195911252890722, "相对强度": 1.4489448612771443, "底分型": 0.6753482684105063, "时间消化": 1.7047252251521656, "量比": 0.6628224353265435, "前低防守距离": 1.3647132593452733, "二次放量启动": 0.2066965710317077, "反弹力度": 0.0009031853141467259},
        "CHOP": {"MACD": 2.20, "相对强度": 1.45, "底分型": 0.68, "时间消化": 1.70, "量比": 0.66, "前低防守距离": 1.36, "二次放量启动": 0.21, "反弹力度": 0.00},
    },
    "三买": {
        "BEAR": {"笔数": 1.95, "量比": 0.96, "MACD": 0.59, "核心验证": 0.20, "底分型": 0.10},
        "BULL": {"MA排列强度": 1.09, "笔数": 0.90, "MACD": 0.72},
        "CHOP": {"回抽深度": 0.6479279421061658, "底分型": 2.992682422467488, "相对强度": 2.344393691634057, "时间消化": 0.5440591640792536, "中枢质量": 0.24757942843710073, "MACD": 1.1613049282390944, "ZG站稳天数": 2.270646025597969, "ATR扩张比": 2.4623588209721676, "突破量持续性": 2.881954914965383},
        "CHOP": {"回抽深度": 0.65, "底分型": 2.99, "相对强度": 2.34, "时间消化": 0.54, "中枢质量": 0.25, "MACD": 1.16, "ZG站稳天数": 2.27, "ATR扩张比": 2.46, "突破量持续性": 2.88},
    },
}

REGIME_THRESHOLDS = {
    "一买": {"BEAR": 95, "BULL": 144, "CHOP": 792},
    "二买": {"BEAR": 187, "BULL": 412, "CHOP": 544},
    "三买": {"BEAR": 90, "BULL": 190, "CHOP": 1000},
}


def _get_weights(buy_type, regime):
    type_weights = REGIME_WEIGHTS.get(buy_type, REGIME_WEIGHTS["一买"])
    return type_weights.get(regime, type_weights.get("CHOP", {}))


class EntryFilter(Buy1ScoringMixin, Buy2ScoringMixin, Buy3ScoringMixin, CommonScoringMixin):
    """L1→L2 入场过滤器：对一买/二买/三买信号做多维打分 + regime阈值过滤。"""

    def __init__(self, code, daily_df, sig_df, regime=None):
        self.code = code
        self.daily_df = daily_df.copy()
        self.daily_df["date"] = pd.to_datetime(self.daily_df["date"])
        self.sig_df = sig_df
        self.regime = regime or "CHOP"
        self.czsc = None  # lazy init
        self.pivots = None
        self.zg = None

    def _init_czsc(self):
        if self.czsc is not None:
            return
        from czsc import CZSC, RawBar, Freq
        ds = self.daily_df.sort_values("date").reset_index(drop=True)
        try:
            bars = [RawBar(symbol=self.code, id=j + 1, dt=r["date"].to_pydatetime(),
                           freq=Freq.D, open=r["open"], close=r["close"],
                           high=r["high"], low=r["low"],
                           vol=r.get("volume", 0), amount=r.get("amount", 0))
                    for j, (_, r) in enumerate(ds.iterrows())]
            self.czsc = CZSC(bars, max_bi_num=50)
        except Exception:
            self.czsc = None
        # Cache pivots
        if self.czsc is not None:
            self.pivots = []
            bl = self.czsc.bi_list
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
                    zl = oh - ol
                    zz = zg - zd if (zg - zd) > 0 else zl
                    self.pivots.append({"dir": "上涨" if b.direction == Direction.Up else "下跌",
                                        "zg": zg, "zd": zd, "gg": gg_val, "dd": dd_val,
                                        "sdt": pb[0].sdt, "edt": pb[-1].edt})
                    i = j
                else:
                    i += 1
            # Cache zg
            up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
            if up_pivots:
                self.zg = up_pivots[-1]["zg"]

    def filter(self, buy_event, regime=None):
        regime = regime or self.regime
        buy_type = self._buy_type_from_event(buy_event)
        self._init_czsc()
        scores = self._score_dimensions(buy_event, buy_type)
        weights = _get_weights(buy_type, regime)
        total = sum(scores.get(k, 0) * weights.get(k, 0) for k in scores)
        threshold = REGIME_THRESHOLDS.get(buy_type, {}).get(regime, 350)
        passed = total >= threshold
        return FilterResult(passed=passed, buy_type=buy_type, total_score=total,
                            dimension_scores=scores,
                            reject_reason="" if passed else f"总分{total:.0f}<{threshold}")

    def _buy_type_from_event(self, buy_event):
        label = str(buy_event.get("signal_label", ""))
        if "一买" in label:
            return "一买"
        if "二买" in label:
            return "二买"
        if "三买" in label:
            return "三买"
        return "一买"

    def _get_entry_price(self, buy_event):
        date = pd.Timestamp(buy_event["date"])
        row = self.daily_df[self.daily_df["date"] >= date]
        if len(row) == 0:
            return 0
        return float(row.iloc[0]["close"])

    def _score_dimensions(self, buy_event, buy_type):
        scores = {}
        signal_date = pd.Timestamp(buy_event["date"])
        sig_row = self.sig_df[self.sig_df["dt"] == signal_date]
        raw_val = ""
        if len(sig_row) > 0:
            raw_val = str(sig_row.iloc[0].get("日线_D1B_BUY1", ""))
            if raw_val in ("", "0"):
                for col in self.sig_df.columns:
                    if "BUY" in col.upper():
                        raw_val = str(sig_row.iloc[0].get(col, ""))
                        if raw_val and raw_val != "0":
                            break
        daily_sliced = self.daily_df[self.daily_df["date"] <= signal_date]
        sig_sliced = self.sig_df[self.sig_df["dt"] <= signal_date]
        entry_price = self._get_entry_price(buy_event)
        rv = raw_val if raw_val else buy_event["signal_label"]

        # 笔数 — 二三买用趋势成熟度，一买不参与
        if buy_type == "三买":
            scores["笔数"] = self._score_stroke_buy3(buy_event["date"])
        elif buy_type == "二买":
            scores["笔数"] = self._score_stroke_buy2(buy_event["date"])

        # 量比
        if buy_type == "二买":
            scores["量比"] = self._score_volume_buy2(buy_event["date"])
        elif buy_type == "三买":
            scores["量比"] = self._score_volume_buy3(buy_event["date"])
        elif buy_type == "一买":
            scores["量比"] = self._score_volume_buy1(buy_event["date"])
        else:
            vol_sliced = daily_sliced.copy()
            if "volume" in vol_sliced.columns and "vol" not in vol_sliced.columns:
                vol_sliced["vol"] = vol_sliced["volume"]
            scores["量比"] = _score_volume(vol_sliced) if len(vol_sliced) > 0 else 0

        # MACD
        scores["MACD"] = _score_macd(sig_sliced) if len(sig_sliced) > 0 else 0
        if scores["MACD"] == 0 and len(sig_sliced) > 0:
            mc_col = next((c for c in sig_sliced.columns if "MACD" in c), None)
            if mc_col:
                for i in range(min(5, len(sig_sliced)) - 1, -1, -1):
                    v = str(sig_sliced.iloc[-i - 1][mc_col])
                    if "金叉" in v:
                        da = min(4, i)
                        scores["MACD"] = 100 if da <= 3 else (60 if da <= 5 else 20)
                        break

        # 核心验证
        if buy_type == "一买":
            scores["核心验证"] = self._score_divergence_power()
        elif buy_type == "二买":
            scores["核心验证"] = self._score_retrace_depth()
        elif buy_type == "三买":
            scores["核心验证"] = self._score_zg_distance_buy3(entry_price)

        # 底分型
        scores["底分型"] = self._score_fractal()

        # 一买特有两维
        if buy_type == "一买":
            scores["中枢"] = self._score_pivot_dist(entry_price)
            scores["距离"] = self._score_zd_distance(entry_price)

        # BULL/BEAR 专属维度
        if self.regime == "BULL":
            if buy_type == "二买":
                scores["MA防守"] = self._score_ma_defense_buy2(buy_event["date"])
            elif buy_type == "三买":
                scores["MA排列强度"] = self._score_ma_alignment_buy3()
        elif self.regime == "BEAR":
            if buy_type == "一买":
                scores["底部放量"] = self._score_bottom_volume_surge_buy1(buy_event["date"])

        # CHOP 专属 4 维 (三买: 回抽深度代替区间位置)
        if self.regime == "CHOP":
            if buy_type == "三买":
                scores["回抽深度"] = self._score_retrace_shallow_buy3(entry_price)
            else:
                scores["区间位置"] = self._score_range_position()
            scores["波动压缩"] = self._score_volatility_compression()
            scores["时间消化"] = self._score_time_digestion(buy_event["date"])
            scores["相对强度"] = self._score_relative_strength()
            if buy_type == "一买":
                scores["前期跌幅"] = self._score_decline_depth_buy1(buy_event["date"])
                scores["成交量萎缩"] = self._score_volume_contraction(buy_event["date"])
            elif buy_type == "二买":
                scores["反弹力度"] = self._score_rebound_strength_buy2(buy_event["date"])
                scores["前低防守距离"] = self._score_prev_low_defense_buy2(buy_event["date"])
                scores["二次放量启动"] = self._score_second_volume_breakout_buy2(buy_event["date"])
            elif buy_type == "三买":
                scores["突破成交量"] = self._score_breakout_volume_buy3(buy_event["date"])
                scores["中枢质量"] = self._score_pivot_quality_buy3()
                scores["ZG站稳天数"] = self._score_zg_stand_days_buy3(entry_price)
                scores["ATR扩张比"] = self._score_breakout_atr_expansion_buy3(buy_event["date"])
                scores["突破量持续性"] = self._score_breakout_volume_persistence_buy3(buy_event["date"])

        return scores

