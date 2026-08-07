"""二买评分维度。"""
import pandas as pd
from pathlib import Path
from czsc.objects import Direction


class Buy2ScoringMixin:
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
            daily = self.daily_df
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
            daily = self.daily_df
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

        def _score_volume_buy2(self, signal_date):
            if self.czsc is None or len(self.czsc.bi_list) < 3: return 0
            bis = self.czsc.bi_list; last3 = bis[-3:]
            if last3[0].direction != Direction.Down or last3[1].direction != Direction.Up: return 0
            H1_date = last3[1].sdt
            T0 = pd.Timestamp(signal_date)
            daily = self.daily_df; daily["date"] = pd.to_datetime(daily["date"])
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

        def _score_ma_defense_buy2(self, signal_date):
            """BULL 二买 MA 防守：入场价相对 MA20/MA60 位置。MA60 上方=趋势完好。"""
            T0 = pd.Timestamp(signal_date)
            daily = self.daily_df
            daily["date"] = pd.to_datetime(daily["date"])
            daily = daily[daily["date"] <= T0]
            if len(daily) < 60:
                return 0
            ma20 = daily["close"].rolling(20).mean().iloc[-1]
            ma60 = daily["close"].rolling(60).mean().iloc[-1]
            price = float(daily["close"].iloc[-1])
            if price <= 0 or ma20 <= 0 or ma60 <= 0:
                return 0
            if price > ma60 and ma20 > ma60:
                return 100
            elif price > ma60 and ma20 <= ma60:
                return 70
            elif price > ma60:
                return 40
            return 0

