"""EntryFilter — L1→L2 入场过滤器。从 backtest.py 分离，独立维护。"""
from collections import namedtuple
import pandas as pd
from pathlib import Path
from czsc.objects import Direction

from zone2_pattern import _score_stroke, _score_volume, _score_macd, _is_bottom_fractal

FilterResult = namedtuple("FilterResult", ["passed", "buy_type", "total_score", "dimension_scores", "reject_reason"])

REGIME_WEIGHTS = {
    "一买": {
        "BEAR": {"核心验证": 1.34, "底分型": 0.43, "MACD": 0.89, "量比": 0.98, "中枢": 0.94, "距离": 0.89},
        "BULL": {"核心验证": 1.21, "底分型": 0.89, "MACD": 0.05, "量比": 0.83, "中枢": 0.94, "距离": 1.87},
        "CHOP": {"核心验证": 0.5877719770252653, "量比": 1.9296510587242894, "MACD": 0.07211783281354563, "底分型": 2.0496247460344184, "中枢": 2.312864421353475, "距离": 1.267394016376168, "区间位置": 1.6079882498225504, "波动压缩": 0.623516358220845, "时间消化": 0.08210690435719653, "相对强度": 0.6620315182610395},
    },
    "二买": {
        "BEAR": {"核心验证": 1.68, "量比": 0.00, "MACD": 1.57, "笔数": 1.00, "底分型": 0.78},
        "BULL": {"核心验证": 2.11, "量比": 0.16, "MACD": 1.38, "笔数": 1.28, "底分型": 1.61},
        "CHOP": {"MACD": 2.195911252890722, "相对强度": 1.4489448612771443, "底分型": 0.6753482684105063, "时间消化": 1.7047252251521656, "量比": 0.6628224353265435, "前低防守距离": 1.3647132593452733, "二次放量启动": 0.2066965710317077, "反弹力度": 0.0009031853141467259},
    },
    "三买": {
        "BEAR": {"核心验证": 1.12, "量比": 0.05, "MACD": 1.08, "笔数": 1.68, "底分型": 1.96},
        "BULL": {"核心验证": 1.07, "量比": 0.26, "MACD": 1.99, "笔数": 1.45, "底分型": 0.48},
        "CHOP": {"回抽深度": 0.6479279421061658, "底分型": 2.992682422467488, "相对强度": 2.344393691634057, "时间消化": 0.5440591640792536, "中枢质量": 0.24757942843710073, "MACD": 1.1613049282390944, "ZG站稳天数": 2.270646025597969, "ATR扩张比": 2.4623588209721676, "突破量持续性": 2.881954914965383},
    },
}

REGIME_THRESHOLDS = {
    "一买": {"BEAR": 255, "BULL": 999, "CHOP": 792},
    "二买": {"BEAR": 273, "BULL": 327, "CHOP": 544},
    "三买": {"BEAR": 999, "BULL": 366, "CHOP": 1000},
}


def _get_weights(buy_type, regime):
    type_weights = REGIME_WEIGHTS.get(buy_type, REGIME_WEIGHTS["一买"])
    return type_weights.get(regime, type_weights.get("CHOP", {}))


class EntryFilter:
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

    # ---- scoring helpers ----

    def _score_zg_stand_days_buy3(self, entry_price):
        """三买 ZG 站稳天数：股价在 ZG 上方持续多少天。越长=突破越牢。"""
        if not self.pivots or entry_price <= 0:
            return 0
        up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
        if not up_pivots:
            return 0
        zg = up_pivots[-1].get("zg", 0)
        if zg <= 0:
            return 0
        daily = self.daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        # Count consecutive days where close > ZG, going backward from latest
        days = 0
        for _, row in daily.iloc[::-1].iterrows():
            if row["close"] > zg:
                days += 1
            else:
                break
        if days >= 10:
            return 100
        elif days >= 6:
            return 80
        elif days >= 3:
            return 55
        elif days >= 1:
            return 25
        return 10

    def _score_breakout_atr_expansion_buy3(self, signal_date):
        """三买突破段 ATR 扩张比：突破后 5 天 ATR / 中枢内 ATR。>1.2=真突破。"""
        if self.czsc is None or not self.pivots:
            return 0
        up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
        if not up_pivots:
            return 0
        last_up = up_pivots[-1]
        daily = self.daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        h, l, c = daily["high"], daily["low"], daily["close"]
        # True Range
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        # 中枢内 ATR(14)
        pivot_mask = (daily["date"] >= last_up["sdt"]) & (daily["date"] <= last_up["edt"])
        pivot_atr = tr[pivot_mask].mean() if pivot_mask.sum() > 0 else 0
        if pivot_atr <= 0:
            return 0
        # 突破后 ATR: 从中枢结束到 signal_date
        T0 = pd.Timestamp(signal_date)
        breakout_mask = (daily["date"] > last_up["edt"]) & (daily["date"] <= T0)
        breakout_atr = tr[breakout_mask].mean() if breakout_mask.sum() > 0 else 0
        if breakout_atr <= 0:
            return 0
        ratio = breakout_atr / pivot_atr
        if ratio > 1.5:
            return 100
        elif ratio > 1.2:
            return 75
        elif ratio > 1.0:
            return 45
        elif ratio > 0.8:
            return 20
        return 10

    def _score_breakout_volume_persistence_buy3(self, signal_date):
        """三买突破量持续性：突破后 5 日平均量 / 中枢内平均量。持续放量=真突破。"""
        if self.czsc is None or not self.pivots:
            return 0
        up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
        if not up_pivots:
            return 0
        last_up = up_pivots[-1]
        T0 = pd.Timestamp(signal_date)
        daily = self.daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        vol_col = "vol" if "vol" in daily.columns else "volume"
        if vol_col not in daily.columns:
            return 0
        # 中枢内均量
        pivot_bars = daily[(daily["date"] >= last_up["sdt"]) & (daily["date"] <= last_up["edt"])]
        if len(pivot_bars) < 3:
            return 0
        pivot_avg_vol = pivot_bars[vol_col].mean()
        if pivot_avg_vol <= 0:
            return 0
        # 突破后 5 日均量
        post_bars = daily[(daily["date"] > last_up["edt"]) & (daily["date"] <= T0)].tail(5)
        if len(post_bars) < 2:
            return 0
        post_avg_vol = post_bars[vol_col].mean()
        if post_avg_vol <= 0:
            return 0
        ratio = post_avg_vol / pivot_avg_vol
        if ratio > 1.5:
            return 100
        elif ratio > 1.2:
            return 70
        elif ratio > 1.0:
            return 40
        elif ratio > 0.8:
            return 15
        return 10


    def _score_prev_low_defense_buy2(self, signal_date):
        """二买前低防守距离：(入场价-一买低点)/一买低点。>0=回踩未破前低=二买有效。"""
        if self.czsc is None or len(self.czsc.bi_list) < 4:
            return 0
        bis = self.czsc.bi_list
        T0 = pd.Timestamp(signal_date)
        prev = [bi for bi in bis if bi.sdt <= T0]
        if len(prev) < 3:
            return 0
        # 核心验证的笔结构是: Down0(一买) -> Up1(反弹) -> Down2(回踩,当前)
        # 前低是一买的低点 = Down0.low
        last_down_bis = [bi for bi in prev if bi.direction == Direction.Down]
        if len(last_down_bis) < 2:
            return 0
        one_buy_low = float(last_down_bis[-2].low)  # 倒数第二个向下笔 = 一买低点
        if one_buy_low <= 0:
            return 0
        # 当前入场价
        T0_ts = pd.Timestamp(signal_date)
        daily = self.daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        row = daily[daily["date"] <= T0_ts]
        if len(row) == 0:
            return 0
        entry_price = float(row["close"].iloc[-1])
        if entry_price <= 0:
            return 0
        dist_pct = (entry_price - one_buy_low) / one_buy_low * 100
        if dist_pct > 8:
            return 100
        elif dist_pct > 5:
            return 85
        elif dist_pct > 3:
            return 65
        elif dist_pct > 1:
            return 40
        elif dist_pct > 0:
            return 15
        return 0

    def _score_second_volume_breakout_buy2(self, signal_date):
        """二买二次放量启动：信号日量 / 回踩期均量。>1.5=放量确认反弹启动。"""
        if self.czsc is None or len(self.czsc.bi_list) < 3:
            return 0
        bis = self.czsc.bi_list
        T0 = pd.Timestamp(signal_date)
        prev = [bi for bi in bis if bi.sdt <= T0]
        if len(prev) < 3:
            return 0
        last3 = prev[-3:]
        if last3[0].direction != Direction.Down:
            return 0
        if last3[1].direction != Direction.Up:
            return 0
        H1_date = last3[1].sdt  # 反弹笔起点 — 回踩期从此开始
        daily = self.daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        vol_col = "vol" if "vol" in daily.columns else "volume"
        if vol_col not in daily.columns:
            return 0
        # 回踩期均量
        retrace = daily[(daily["date"] >= H1_date) & (daily["date"] <= T0)]
        if len(retrace) < 3:
            return 0
        retrace_avg = retrace[vol_col].mean()
        if retrace_avg <= 0:
            return 0
        # 信号日（或最后一日）量 — 测量回踩结束时的量能启动
        signal_vol = float(daily[daily["date"] <= T0][vol_col].iloc[-1])
        ratio = signal_vol / retrace_avg
        if ratio > 2.0:
            return 100
        elif ratio > 1.5:
            return 75
        elif ratio > 1.2:
            return 45
        elif ratio > 1.0:
            return 20
        return 10

    def _score_divergence_power(self):
        if self.czsc is None: return 0
        dbs = [bi for bi in self.czsc.bi_list if bi.direction == Direction.Down]
        if len(dbs) >= 2: a, b = dbs[-2], dbs[-1]
        else: return 0
        if a.power <= 0: return 0
        ratio = b.power / a.power
        if ratio < 0.3: return 100
        elif ratio < 0.5: return 85
        elif ratio < 0.7: return 65
        elif ratio < 0.9: return 40
        elif ratio < 1.1: return 15
        return 0

    def _score_retrace_depth(self):
        """二买核心验证：回踩深度。浅回踩=一买有效=高分。"""
        if self.czsc is None or len(self.czsc.bi_list) < 3: return 0
        bis = self.czsc.bi_list; last3 = bis[-3:]
        if last3[0].direction != Direction.Down: return 0
        if last3[1].direction != Direction.Up: return 0
        if last3[2].direction != Direction.Down: return 0
        L1, H1, L2 = float(last3[0].low), float(last3[1].high), float(last3[2].low)
        if H1 <= L1: return 0
        ratio = (H1 - L2) / (H1 - L1)
        # 放宽阈值：浅回踩(<0.3)和中等回踩(0.3-0.6)都给较高分
        if ratio < 0.0: return 0
        if ratio < 0.3: return 100
        elif ratio < 0.5: return 80
        elif ratio < 0.7: return 60
        elif ratio < 1.0: return 35
        elif ratio < 1.3: return 10
        return 5

    def _score_fractal(self):
        if self.czsc is None or not hasattr(self.czsc, "fx_list") or not self.czsc.fx_list: return 0
        btm_fx = None
        for fx in reversed(self.czsc.fx_list):
            if _is_bottom_fractal(fx): btm_fx = fx; break
        if btm_fx is None: return 0
        lf = btm_fx
        kc = len(lf.raw_bars) if lf.raw_bars and len(lf.raw_bars) >= 3 else 3
        if kc >= 10: sk = 100
        elif kc >= 8: sk = 85
        elif kc >= 6: sk = 70
        elif kc >= 5: sk = 55
        elif kc >= 4: sk = 40
        else: sk = 20
        mid = len(lf.raw_bars) // 2
        first_bar = lf.raw_bars[0]; last_bar = lf.raw_bars[-1]
        prev_bar = lf.raw_bars[-2] if len(lf.raw_bars) >= 3 else None
        first_body_top = max(first_bar.open, first_bar.close)
        first_body_bot = min(first_bar.open, first_bar.close)
        first_body_mid = (first_body_top + first_body_bot) / 2
        gap_up = last_bar.close > last_bar.open
        if prev_bar is not None: gap_up = gap_up and last_bar.open > prev_bar.close
        if last_bar.close > first_bar.high: sc = 60
        elif last_bar.close > first_body_mid and gap_up: sc = 45
        elif last_bar.close > first_body_mid: sc = 30
        else: sc = 8
        down_bis = [bi for bi in self.czsc.finished_bis if bi.direction == Direction.Down]
        if down_bis:
            last_down = down_bis[-1]
            if last_down.low > 0:
                bottom_low = lf.raw_bars[mid].low
                dist_pct = abs(bottom_low - last_down.low) / last_down.low * 100
                if dist_pct <= 0.5: sc += 20
                elif dist_pct <= 1.0: sc += 13
                elif dist_pct <= 2.0: sc += 7
        return min(sk + sc, 100)

    def _score_zg_distance_buy3(self, entry_price):
        if not self.pivots or entry_price <= 0: return 0
        up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
        if not up_pivots: return 0
        zg = up_pivots[-1].get("zg", 0)
        if zg <= 0: return 0
        dist = (entry_price - zg) / zg * 100
        if dist <= 1: return 10
        elif dist <= 3: return 50
        elif dist <= 5: return 100
        elif dist <= 8: return 80
        elif dist <= 12: return 50
        return 10

    def _score_volume_buy1(self, signal_date):
        """一买量比：底部缩量=抛压耗尽=高分。ratio < 0.3 满分。"""
        T0 = pd.Timestamp(signal_date)
        daily = self.daily_df.copy(); daily["date"] = pd.to_datetime(daily["date"])
        daily = daily[daily["date"] <= T0]
        if len(daily) < 20: return 0
        vol_col = "vol" if "vol" in daily.columns else "volume"
        if vol_col not in daily.columns: return 0
        current_vol = daily[vol_col].iloc[-1]
        ma20_vol = daily[vol_col].tail(20).mean()
        if ma20_vol <= 0: return 0
        ratio = current_vol / ma20_vol
        if ratio < 0.3: return 100
        elif ratio < 0.5: return 85
        elif ratio < 0.7: return 60
        elif ratio < 1.0: return 30
        return 10

    def _score_volume_buy2(self, signal_date):
        if self.czsc is None or len(self.czsc.bi_list) < 3: return 0
        bis = self.czsc.bi_list; last3 = bis[-3:]
        if last3[0].direction != Direction.Down or last3[1].direction != Direction.Up: return 0
        H1_date = last3[1].sdt
        T0 = pd.Timestamp(signal_date)
        daily = self.daily_df.copy(); daily["date"] = pd.to_datetime(daily["date"])
        retrace = daily[(daily["date"] >= H1_date) & (daily["date"] <= T0)]
        if len(retrace) == 0: return 0
        vol_col = "vol" if "vol" in retrace.columns else "volume"
        if vol_col not in retrace.columns: return 0
        retrace_avg = retrace[vol_col].mean()
        ma20_vol = daily[vol_col].tail(20).mean()
        if ma20_vol <= 0: return 0
        ratio = retrace_avg / ma20_vol
        if ratio < 0.3: return 100
        elif ratio < 0.5: return 75
        elif ratio < 0.7: return 50
        elif ratio < 1.0: return 25
        return 10

    def _score_volume_buy3(self, signal_date):
        """三买量比：离开中枢段的平均量 / 20日均量。缩量回抽 = 好。"""
        if self.czsc is None: return 0
        T0 = pd.Timestamp(signal_date)
        daily = self.daily_df.copy(); daily["date"] = pd.to_datetime(daily["date"])
        vol_col = "vol" if "vol" in daily.columns else "volume"
        if vol_col not in daily.columns: return 0
        # 找最后上涨中枢的 edt
        if self.pivots:
            up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
            if up_pivots:
                last_up = up_pivots[-1]
                H_out_date = last_up.get("edt")
            else:
                H_out_date = None
        else:
            H_out_date = None
        # 如果没有中枢数据，直接用 T0 前 20 天作为回抽窗口
        if H_out_date is None:
            H_out_date = T0 - pd.Timedelta(days=20)
        retrace = daily[(daily["date"] >= H_out_date) & (daily["date"] <= T0)]
        if len(retrace) == 0: return 0
        retrace_avg = retrace[vol_col].mean()
        ma20_vol = daily[vol_col].tail(20).mean()
        if ma20_vol <= 0: return 0
        ratio = retrace_avg / ma20_vol
        if ratio < 0.3: return 100
        elif ratio < 0.6: return 80
        elif ratio < 0.8: return 60
        elif ratio < 1.0: return 40
        return 15

    def _score_rebound_strength_buy2(self, signal_date):
        """二买反弹力度：一买低点→反弹高点涨幅%。越大=反转越强。"""
        if self.czsc is None or len(self.czsc.bi_list) < 3: return 0
        bis = self.czsc.bi_list
        T0 = pd.Timestamp(signal_date)
        # Find the last down-up-down pattern before T0
        prev = [bi for bi in bis if bi.sdt <= T0]
        if len(prev) < 3: return 0
        last3 = prev[-3:]
        if last3[0].direction != Direction.Down: return 0
        if last3[1].direction != Direction.Up: return 0
        L1 = float(last3[0].low); H1 = float(last3[1].high)
        if L1 <= 0: return 0
        rebound = (H1 - L1) / L1 * 100
        if rebound > 30: return 100
        elif rebound > 20: return 80
        elif rebound > 12: return 60
        elif rebound > 6: return 30
        return 10

    def _score_stroke_buy2(self, signal_date):
        """二买笔数: 一买→反弹→回踩周期中的趋势成熟度。"""
        if self.czsc is None: return 0
        bis = self.czsc.bi_list
        if len(bis) < 3: return 0
        T0 = pd.Timestamp(signal_date)
        # Count upward bis in the recent cycle
        ups = [bi for bi in bis if bi.direction == Direction.Up and bi.sdt <= T0]
        if len(ups) < 2: return 0
        downs = [bi for bi in bis if bi.direction == Direction.Down and bi.sdt <= T0]
        if not downs: return 0
        # Count bis from last significant down to T0
        last_down = downs[-1]
        count = sum(1 for bi in bis if bi.sdt >= last_down.sdt and bi.sdt <= T0)
        if count >= 10: return 100
        elif count >= 8: return 80
        elif count >= 6: return 60
        elif count >= 4: return 40
        elif count >= 2: return 20
        return 10

    def _score_stroke_buy3(self, signal_date):
        """三买笔数: 中枢前上涨趋势成熟度。上游向上笔越多=趋势越稳固。"""
        if self.czsc is None: return 0
        bis = self.czsc.bi_list
        if len(bis) < 3: return 0
        T0 = pd.Timestamp(signal_date)
        ups = [bi for bi in bis if bi.direction == Direction.Up and bi.sdt <= T0]
        if len(ups) < 2: return 0
        if not self.pivots:
            count = len(ups)
        else:
            last_up_pivot = None
            for p in reversed(self.pivots):
                if p["dir"] == "上涨":
                    last_up_pivot = p; break
            count = sum(1 for bi in ups if bi.sdt >= last_up_pivot["sdt"]) if last_up_pivot else len(ups)
        if count >= 8: return 100
        elif count >= 6: return 80
        elif count >= 5: return 60
        elif count >= 4: return 40
        elif count >= 3: return 20
        return 10

    def _score_breakout_volume_buy3(self, signal_date):
        """三买突破成交量：离开中枢笔的量 / 中枢内均量。比率>1.5=真突破。"""
        if self.czsc is None or not self.pivots: return 0
        up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
        if not up_pivots: return 0
        last_up = up_pivots[-1]
        leaving_bi = None
        for bi in self.czsc.bi_list:
            if bi.sdt > last_up["edt"]:
                leaving_bi = bi; break
        if leaving_bi is None: return 0
        T0 = pd.Timestamp(signal_date)
        daily = self.daily_df.copy(); daily["date"] = pd.to_datetime(daily["date"])
        vol_col = "vol" if "vol" in daily.columns else "volume"
        if vol_col not in daily.columns: return 0
        # Leaving bi volume
        leaving_bars = daily[(daily["date"] >= leaving_bi.sdt) & (daily["date"] <= leaving_bi.edt)]
        if len(leaving_bars) == 0: return 0
        leaving_vol = leaving_bars[vol_col].mean()
        # Pivot area volume
        pivot_bars = daily[(daily["date"] >= last_up["sdt"]) & (daily["date"] <= last_up["edt"])]
        if len(pivot_bars) == 0: return 0
        pivot_vol = pivot_bars[vol_col].mean()
        if pivot_vol <= 0: return 0
        ratio = leaving_vol / pivot_vol
        if ratio > 2.0: return 100
        elif ratio > 1.5: return 80
        elif ratio > 1.2: return 50
        elif ratio > 1.0: return 20
        return 10

    def _score_pivot_quality_buy3(self):
        """三买中枢质量：中枢内笔数越多=结构越扎实。≥6笔满分。"""
        if self.czsc is None or not self.pivots: return 0
        up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
        if not up_pivots: return 0
        last_up = up_pivots[-1]
        # Count bis within this pivot
        count = sum(1 for bi in self.czsc.bi_list 
                    if bi.sdt >= last_up["sdt"] and bi.edt <= last_up["edt"])
        if count >= 8: return 100
        elif count >= 6: return 80
        elif count >= 4: return 50
        elif count >= 3: return 20
        return 10

    def _score_retrace_shallow_buy3(self, entry_price):
        """三买回抽深度: (ZG-入场价)/ZG，越小=回抽越浅=越强。"""
        if not self.pivots or entry_price <= 0: return 0
        up_pivots = [p for p in self.pivots if p["dir"] == "上涨"]
        if not up_pivots: return 0
        zg = up_pivots[-1].get("zg", 0)
        if zg <= 0: return 0
        depth = (zg - entry_price) / zg * 100
        # 回抽浅=好, 深=差 (回抽指 entry_price 低于 ZG 的部分)
        if depth <= 0: return 100
        elif depth <= 2: return 80
        elif depth <= 5: return 50
        elif depth <= 8: return 20
        return 10

    def _score_decline_depth_buy1(self, signal_date):
        """一买前期跌幅：从最后向上笔高点回落幅度。20-40%=最佳。"""
        if self.czsc is None: return 0
        ups = [bi for bi in self.czsc.bi_list if bi.direction == Direction.Up and bi.sdt <= pd.Timestamp(signal_date)]
        if len(ups) < 2: return 0
        recent_high = max(bi.high for bi in ups[-3:])
        T0 = pd.Timestamp(signal_date)
        daily_slice = self.daily_df[self.daily_df["date"] <= T0]
        if len(daily_slice) == 0: return 0
        entry_price = float(daily_slice["close"].iloc[-1])
        if recent_high <= 0 or entry_price <= 0: return 0
        decline = (recent_high - entry_price) / recent_high * 100
        if decline < 5: return 10
        elif decline < 15: return 50
        elif decline < 25: return 100
        elif decline < 35: return 80
        elif decline < 50: return 40
        return 10

    def _score_volume_contraction(self, signal_date):
        """一买成交量萎缩度：下跌末期量缩=抛压耗尽。后半段量/前半段量，比值越小越好。"""
        T0 = pd.Timestamp(signal_date)
        daily = self.daily_df.copy(); daily["date"] = pd.to_datetime(daily["date"])
        daily = daily[daily["date"] <= T0]
        if len(daily) < 40: return 0
        vol_col = "vol" if "vol" in daily.columns else "volume"
        if vol_col not in daily.columns: return 0
        # Split recent 40 days into two halves
        early = daily[vol_col].tail(40).head(20).mean()
        late = daily[vol_col].tail(20).mean()
        if early <= 0: return 0
        ratio = late / early
        if ratio < 0.4: return 100
        elif ratio < 0.6: return 80
        elif ratio < 0.8: return 50
        elif ratio < 1.0: return 20
        return 10

    def _score_range_position(self):
        """CHOP 震荡区间位置：买在底部=高分。60日区间底部30%得分最高。"""
        daily = self.daily_df
        if len(daily) < 60: return 0
        high60 = daily['high'].tail(60).max()
        low60 = daily['low'].tail(60).min()
        if high60 <= low60: return 0
        pos = (daily['close'].iloc[-1] - low60) / (high60 - low60)
        if pos <= 0.30: return 70 + (0.30 - pos) / 0.30 * 30
        elif pos <= 0.70: return (0.70 - pos) / 0.40 * 30
        return 0

    def _score_volatility_compression(self):
        """CHOP 波动压缩：ATR(14)/ATR(50) 缩小=突破前兆。"""
        daily = self.daily_df
        if len(daily) < 50: return 0
        h, l, c = daily['high'], daily['low'], daily['close']
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        atr50 = tr.rolling(50).mean().iloc[-1]
        if atr50 <= 0: return 0
        ratio = atr14 / atr50
        if ratio < 0.5: return 100
        elif ratio < 0.7: return 80
        elif ratio < 0.9: return 40
        elif ratio < 1.0: return 20
        return 0

    def _score_time_digestion(self, signal_date):
        """CHOP 时间消化：距最后向下笔完成越久=整理越充分。"""
        if self.czsc is None: return 0
        downs = [bi for bi in self.czsc.bi_list if bi.direction == Direction.Down]
        if not downs: return 0
        last_edt = downs[-1].edt
        T0 = pd.Timestamp(signal_date)
        days = abs((T0 - last_edt).days)  # 可能 edt > T0
        if days < 5: return 10
        if days > 180: return 100
        return min(100, int(days / 180 * 100))

    def _score_relative_strength(self):
        """CHOP 相对强度：个股20日涨幅 vs 上证20日涨幅。"""
        daily = self.daily_df
        if len(daily) < 20: return 0
        stock_ret = (daily['close'].iloc[-1] / daily['close'].iloc[-21] - 1) * 100
        # 上证指数数据
        try:
            proxy = pd.read_parquet(Path("/Users/hz/Desktop/Algorithmic Trading Platform") / "data/index/000001.parquet")
            proxy = proxy.sort_values('date')
            idx_ret = (proxy['close'].iloc[-1] / proxy['close'].iloc[-21] - 1) * 100
        except:
            idx_ret = 0
        excess = stock_ret - idx_ret
        if excess > 5: return 100
        elif excess > 2: return 70
        elif excess > -2: return 40
        elif excess > -5: return 20
        return 0

    def _score_pivot_dist(self, price):
        """一买中枢距离：距最后下跌笔低点越远=反弹确认越强。线性映射0-18%到10-100分。"""
        if self.czsc is None or price <= 0: return 0
        downs = [bi for bi in self.czsc.bi_list if bi.direction == Direction.Down]
        if not downs: return 0
        nearest_low = downs[-1].low
        if nearest_low <= 0: return 0
        dist = (price - nearest_low) / nearest_low * 100
        return min(100, max(10, int(dist / 18 * 90 + 10)))

    def _score_zd_distance(self, price):
        """一买距下跌中枢下沿：越远=反弹空间越大。线性映射5-30%到10-100分。"""
        if not self.pivots or price <= 0: return 0
        down_pivots = [p for p in self.pivots if p["dir"] == "下跌"]
        if not down_pivots: return 0
        zd = down_pivots[-1].get("zd", 0)
        if zd <= 0: return 0
        dist = (price - zd) / zd * 100
        return min(100, max(10, int(dist / 30 * 90 + 10)))
