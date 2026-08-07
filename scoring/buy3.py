"""三买评分维度。"""
import pandas as pd
from pathlib import Path
from czsc.objects import Direction


class Buy3ScoringMixin:
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
            daily = self.daily_df
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
            daily = self.daily_df
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
            daily = self.daily_df
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

        def _score_volume_buy3(self, signal_date):
            """三买量比：离开中枢段的平均量 / 20日均量。缩量回抽 = 好。"""
            if self.czsc is None: return 0
            T0 = pd.Timestamp(signal_date)
            daily = self.daily_df; daily["date"] = pd.to_datetime(daily["date"])
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
            daily = self.daily_df; daily["date"] = pd.to_datetime(daily["date"])
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

        def _score_ma_alignment_buy3(self):
            """BULL 三买 MA 多头排列强度：MA5>MA20>MA60 且三线斜率>0。"""
            daily = self.daily_df
            daily["date"] = pd.to_datetime(daily["date"])
            if len(daily) < 60:
                return 0
            ma5 = daily["close"].rolling(5).mean()
            ma20 = daily["close"].rolling(20).mean()
            ma60 = daily["close"].rolling(60).mean()
            v5, v20, v60 = float(ma5.iloc[-1]), float(ma20.iloc[-1]), float(ma60.iloc[-1])
            if v5 <= 0 or v20 <= 0 or v60 <= 0:
                return 0
            # 斜率（5日变动）
            s5 = (ma5.iloc[-1] - ma5.iloc[-6]) / ma5.iloc[-6] if len(ma5) >= 6 else 0
            s20 = (ma20.iloc[-1] - ma20.iloc[-21]) / ma20.iloc[-21] if len(ma20) >= 21 else 0
            s60 = (ma60.iloc[-1] - ma60.iloc[-61]) / ma60.iloc[-61] if len(ma60) >= 61 else 0
            if v5 > v20 > v60 and s5 > 0 and s20 > 0 and s60 > 0:
                return 100
            elif v5 > v20 > v60 and s5 > 0:
                return 70
            elif v5 > v20 > v60:
                return 40
            return 0

