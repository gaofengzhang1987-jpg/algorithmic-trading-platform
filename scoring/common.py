"""共享评分维度 — 跨买点类型使用。"""
import pandas as pd
from pathlib import Path
from czsc.objects import Direction
from zone2_pattern import _is_bottom_fractal


class CommonScoringMixin:
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
                proxy = pd.read_parquet(BASE_DIR / "data/index/000001.parquet")
                proxy = proxy.sort_values('date')
                idx_ret = (proxy['close'].iloc[-1] / proxy['close'].iloc[-21] - 1) * 100
            except Exception:
                idx_ret = 0  # fallback: 无上证指数数据时默认与指数同步
            excess = stock_ret - idx_ret
            if excess > 5: return 100
            elif excess > 2: return 70
            elif excess > -2: return 40
            elif excess > -5: return 20
            return 0

