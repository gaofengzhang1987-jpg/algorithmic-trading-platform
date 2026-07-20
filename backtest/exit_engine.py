"""缠论结构出场引擎 — 运作仓位状态机的信号判断器。"""
import pandas as pd
from pathlib import Path
from czsc import CZSC, RawBar, Freq
from czsc.objects import Direction
from core.constants import (
    DISCOUNT_BUY1, DISCOUNT_BUY2, DISCOUNT_BUY3,
    STOP_LOSS_PCT, HALF_CUT_TIMEOUT, BASE_DIR,
)


def _load_min30_bars(code):
    p30 = BASE_DIR / "data" / "min30" / f"{code}.parquet"
    if not p30.exists():
        return None
    return pd.read_parquet(p30)


class ExitEngine:
    """缠论结构出场引擎 + 仓位状态机。

    出场优先级（每 bar，最先触发胜出）：
      1. 结构止损：close <= defense -> 全仓退出
      2. V 型暴跌：close < 最后上涨中枢 GG -> 全仓退出
      3. FULL -> HALF（顶分型 + 力度背驰 + 创新高）
      4. HALF -> 回补 / 二卖确认 / 30 天超时
      5. 卖点信号：次日开盘价退出
    """

    def __init__(self, code, entry_price, entry_date, buy_type, struct_df):
        self.code = code
        self.entry_price = entry_price
        self.entry_date = pd.Timestamp(entry_date)
        self.buy_type = buy_type
        self.struct_df = struct_df
        self.entry_bi_low_1buy = 0.0
        self.entry_bi_low_2buy = 0.0
        self.entry_bi_low_3buy = 0.0
        self.entry_pivot_gg = 0.0
        self._snapshot_entry_structure()
        self.defense = self._compute_init_stop()

    def _snapshot_entry_structure(self):
        if self.struct_df is None or self.struct_df.empty:
            return
        sdf = self.struct_df
        entry_d = self.entry_date.date()

        down_bis = sdf[
            (sdf["direction"].str.contains("向下", na=False)) &
            (pd.to_datetime(sdf["edt"]).dt.date <= entry_d)
        ]
        if len(down_bis) > 0:
            last_down = down_bis.iloc[-1]
            self.entry_bi_low_1buy = float(last_down["low"])
            if "底分型" in str(last_down["fx_b_mark"]):
                self.entry_bi_low_1buy = float(last_down["fx_b_low"])

        fx_before_entry = []
        for _, row in sdf.iterrows():
            edt_ts = pd.to_datetime(row["edt"]); edt_date = edt_ts.date()
            if edt_date <= entry_d and "底分型" in str(row["fx_b_mark"]):
                fx_before_entry.append(float(row["fx_b_low"]))
        if fx_before_entry:
            self.entry_bi_low_2buy = fx_before_entry[-1]
        self.entry_bi_low_3buy = self.entry_bi_low_2buy

        up_pivots = sdf[
            (sdf["pivot_dir"] == "上涨") &
            (pd.to_datetime(sdf["sdt"]).dt.date <= entry_d)
        ]
        if len(up_pivots) > 0:
            self.entry_pivot_gg = float(up_pivots.iloc[-1]["pivot_gg"])

    def _compute_init_stop(self):
        if "一买" in str(self.buy_type):
            low_val = self.entry_bi_low_1buy
            if low_val <= 0:
                low_val = self.entry_bi_low_2buy
            if low_val <= 0:
                low_val = self.entry_price * DISCOUNT_BUY2
            return low_val * DISCOUNT_BUY1
        elif "二买" in str(self.buy_type):
            v1 = self.entry_bi_low_1buy if self.entry_bi_low_1buy > 0 else 0
            v2 = self.entry_bi_low_2buy if self.entry_bi_low_2buy > 0 else v1
            base = max(v1, v2)
            if base <= 0:
                base = self.entry_price * DISCOUNT_BUY2
            return base * DISCOUNT_BUY2
        elif "三买" in str(self.buy_type):
            gg = self.entry_pivot_gg if self.entry_pivot_gg > 0 else 0
            v3 = self.entry_bi_low_3buy if self.entry_bi_low_3buy > 0 else 0
            base = max(gg, v3)
            if base <= 0:
                base = self.entry_price * DISCOUNT_BUY3
            return base * DISCOUNT_BUY3
        return self.entry_price * DISCOUNT_BUY2

    def update_defense(self, bar_date):
        if self.struct_df is None or self.struct_df.empty:
            return
        bar_d = bar_date.date() if hasattr(bar_date, 'date') else bar_date
        if isinstance(bar_d, pd.Timestamp):
            bar_d = bar_d.date()
        sdf = self.struct_df

        if "一买" in str(self.buy_type) or "二买" in str(self.buy_type):
            up_pivots = sdf[
                (sdf["pivot_dir"] == "上涨") &
                (pd.to_datetime(sdf["sdt"]).dt.date > self.entry_date.date()) &
                (pd.to_datetime(sdf["sdt"]).dt.date <= bar_d)
            ]
            if len(up_pivots) > 0:
                new_gg = float(up_pivots.iloc[-1]["pivot_gg"])
                if new_gg > self.defense:
                    self.defense = new_gg

        if "一买" in str(self.buy_type):
            new_2buy = sdf[
                (sdf["direction"].str.contains("向下", na=False)) &
                (pd.to_datetime(sdf["edt"]).dt.date > self.entry_date.date()) &
                (pd.to_datetime(sdf["edt"]).dt.date <= bar_d)
            ]
            if len(new_2buy) > 0:
                last_down = new_2buy.iloc[-1]
                fx_low = float(last_down["fx_b_low"]) if "底分型" in str(last_down["fx_b_mark"]) else float(last_down["low"])
                if fx_low > self.defense:
                    self.defense = fx_low

    def check_divergence(self, bar_date):
        bar_d = bar_date.date() if hasattr(bar_date, 'date') else bar_date
        if isinstance(bar_d, pd.Timestamp):
            bar_d = bar_d.date()

        min30 = _load_min30_bars(self.code)
        if min30 is not None and len(min30) > 0:
            result = self._check_divergence_level(min30, Freq.F30, bar_d)
            if result is not None:
                return result

        if self.struct_df is not None and not self.struct_df.empty:
            return self._check_divergence_from_struct(bar_d)

        return False

    def _check_divergence_level(self, df, freq, bar_date):
        try:
            df_tail = df[df["date"] <= pd.Timestamp(bar_date)].tail(2000).reset_index(drop=True)
            if len(df_tail) < 30:
                return None
            bars = [RawBar(symbol=self.code, id=j+1, dt=r["date"].to_pydatetime(),
                           freq=freq, open=r["open"], close=r["close"],
                           high=r["high"], low=r["low"],
                           vol=r.get("volume", 0), amount=r.get("amount", 0))
                    for j, (_, r) in enumerate(df_tail.iterrows())]
            c = CZSC(bars, max_bi_num=50)
            up_bis = [bi for bi in c.bi_list if bi.direction == Direction.Up]
            up_no_zs = [bi for bi in up_bis if not bi.fx_b.has_zs]
            if len(up_no_zs) >= 2:
                a, b = up_no_zs[-2], up_no_zs[-1]
                return b.power < a.power
            elif len(up_bis) >= 2:
                a, b = up_bis[-2], up_bis[-1]
                return b.power < a.power
        except Exception:
            pass
        return None

    def _check_divergence_from_struct(self, bar_date):
        sdf = self.struct_df
        up_bis = sdf[
            (sdf["direction"].str.contains("向上", na=False)) &
            (pd.to_datetime(sdf["edt"]).dt.date <= bar_date) &
            (sdf["pivot_id"] == -1)
        ]
        if len(up_bis) < 2:
            up_bis = sdf[
                (sdf["direction"].str.contains("向上", na=False)) &
                (pd.to_datetime(sdf["edt"]).dt.date <= bar_date)
            ]
        if len(up_bis) >= 2:
            a, b = up_bis.iloc[-2], up_bis.iloc[-1]
            return float(b["power"]) < float(a["power"])
        return False

    def get_prev_same_dir_high(self, bar_date):
        if self.struct_df is None or self.struct_df.empty:
            return self.entry_price
        sdf = self.struct_df
        bar_d = bar_date.date() if hasattr(bar_date, 'date') else bar_date
        if isinstance(bar_d, pd.Timestamp):
            bar_d = bar_d.date()
        up_bis = sdf[
            (sdf["direction"].str.contains("向上", na=False)) &
            (pd.to_datetime(sdf["edt"]).dt.date < bar_d)
        ]
        if len(up_bis) >= 2:
            return float(up_bis.iloc[-2]["high"])
        elif len(up_bis) == 1:
            return float(up_bis.iloc[-1]["high"])
        return self.entry_price

    def check_half_cut(self, bar_date, bar_high, fx_on_bar):
        if fx_on_bar is None:
            return None
        if "顶分型" not in str(fx_on_bar.get("mark", "")):
            return None
        prev_high = self.get_prev_same_dir_high(bar_date)
        if bar_high <= prev_high:
            return None
        if not self.check_divergence(bar_date):
            return None
        return {"type": "half_cut", "date": bar_date, "fx_high": fx_on_bar.get("high", bar_high)}

    def check_buyback(self, bar_date, bar_high, half_cut_fx_high):
        return bar_high > half_cut_fx_high

    def check_second_sell(self, bar_date, bar_high, half_cut_fx_high, fx_on_bar):
        if fx_on_bar is None:
            return False
        if "顶分型" not in str(fx_on_bar.get("mark", "")):
            return False
        return True

    def check_v_drop(self, bar_date, bar_close):
        if self.struct_df is None or self.struct_df.empty:
            return False
        sdf = self.struct_df
        bar_d = bar_date.date() if hasattr(bar_date, 'date') else bar_date
        if isinstance(bar_d, pd.Timestamp):
            bar_d = bar_d.date()
        up_pivots = sdf[
            (sdf["pivot_dir"] == "上涨") &
            (pd.to_datetime(sdf["sdt"]).dt.date > self.entry_date.date()) &
            (pd.to_datetime(sdf["edt"]).dt.date <= bar_d)
        ]
        if len(up_pivots) > 0:
            last_gg = float(up_pivots.iloc[-1]["pivot_gg"])
            if last_gg > 0 and last_gg < bar_close * 1.5:
                return bar_close < last_gg
        return False

    def rebind_defense(self, bar_date):
        if self.struct_df is None:
            return
        bar_d = bar_date.date() if hasattr(bar_date, 'date') else bar_date
        if isinstance(bar_d, pd.Timestamp):
            bar_d = bar_d.date()
        sdf = self.struct_df
        up_pivots = sdf[
            (sdf["pivot_dir"] == "上涨") &
            (pd.to_datetime(sdf["edt"]).dt.date <= bar_d)
        ]
        if len(up_pivots) > 0:
            self.entry_pivot_gg = float(up_pivots.iloc[-1]["pivot_gg"])
        down_bis = sdf[
            (sdf["direction"].str.contains("向下", na=False)) &
            (pd.to_datetime(sdf["edt"]).dt.date <= bar_d)
        ]
        if len(down_bis) > 0:
            last_down = down_bis.iloc[-1]
            self.entry_bi_low_1buy = float(last_down["low"])
        cur = self._compute_init_stop()
        if cur > self.defense:
            self.defense = cur
        self.update_defense(pd.Timestamp(bar_d))
