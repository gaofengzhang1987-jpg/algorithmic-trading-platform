"""一买评分维度。"""
import pandas as pd
from pathlib import Path
from czsc.objects import Direction


class Buy1ScoringMixin:
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

        def _score_volume_buy1(self, signal_date):
            """一买量比：底部缩量=抛压耗尽=高分。ratio < 0.3 满分。"""
            T0 = pd.Timestamp(signal_date)
            daily = self.daily_df; daily["date"] = pd.to_datetime(daily["date"])
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

        def _score_bottom_volume_surge_buy1(self, signal_date):
            """BEAR 一买底部放量：信号日+前2日平均量 / 前20日均量。放量=恐慌出清=真底。"""
            T0 = pd.Timestamp(signal_date)
            daily = self.daily_df
            daily["date"] = pd.to_datetime(daily["date"])
            daily = daily[daily["date"] <= T0]
            if len(daily) < 20:
                return 0
            vol_col = "vol" if "vol" in daily.columns else "volume"
            if vol_col not in daily.columns:
                return 0
            recent_avg = daily[vol_col].tail(3).mean()
            ma20_vol = daily[vol_col].tail(20).mean()
            if ma20_vol <= 0:
                return 0
            ratio = recent_avg / ma20_vol
            if ratio > 2.0:
                return 100
            elif ratio > 1.5:
                return 75
            elif ratio > 1.2:
                return 50
            elif ratio > 1.0:
                return 20
            return 0

        def _score_volume_contraction(self, signal_date):
            """一买成交量萎缩度：下跌末期量缩=抛压耗尽。后半段量/前半段量，比值越小越好。"""
            T0 = pd.Timestamp(signal_date)
            daily = self.daily_df; daily["date"] = pd.to_datetime(daily["date"])
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

