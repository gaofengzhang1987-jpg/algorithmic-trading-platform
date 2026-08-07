"""缠论结构出场引擎 — 运作仓位状态机的信号判断器。"""
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from core.constants import (
    DISCOUNT_BUY1, DISCOUNT_BUY2, DISCOUNT_BUY3,
    STOP_LOSS_PCT, HALF_CUT_TIMEOUT, BASE_DIR,
)


@dataclass
class BarResult:
    """Single-bar exit result (B2: unified return type)."""
    state: str
    position_pct: float
    exit: bool
    exit_reason: str | None = None
    partial_exits: list = None  # [(pct, price)] for weighted return
    trajectory: list = None


def _load_struct_30m(code):
    p30 = BASE_DIR / "data" / "struct_cache_30m" / f"{code}.parquet"
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

    def __init__(self, code, entry_price, entry_date, buy_type, struct_df, trajectory_log: bool = False):
        self.code = code
        self.entry_price = entry_price
        self.entry_date = pd.Timestamp(entry_date)
        self.buy_type = buy_type
        self.struct_df = struct_df
        self.struct_30m = _load_struct_30m(code)
        self.entry_bi_low_1buy = 0.0
        self.entry_bi_low_2buy = 0.0
        self.entry_bi_low_3buy = 0.0
        self.entry_pivot_gg = 0.0
        self._snapshot_entry_structure()
        self.defense = self._compute_init_stop()
        # State machine (B1: internalized from runner)
        self.state = "FULL"
        self.half_cut_fx_high = 0.0
        self.half_cut_day_count = 0
        self.position_pct = 1.0
        self._high_since_half_cut = 0.0
        self._partial_exits = []  # [(pct, price)] partial exit records
        self.trajectory_log = trajectory_log
        self._trajectory: list[dict] = []

    def _log_trajectory(self, event: str, bar_date, bar_close: float, detail: str = ""):
        if not self.trajectory_log:
            return
        self._trajectory.append({
            "date": str(pd.Timestamp(bar_date).date()),
            "event": event,
            "price": round(bar_close, 2),
            "defense": round(self.defense, 2) if hasattr(self, "defense") and self.defense else 0,
            "state": self.state,
            "detail": detail,
        })

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

        fx_rows = sdf[(pd.to_datetime(sdf["edt"]).dt.date <= entry_d) &
                      (sdf["fx_b_mark"].str.contains("底分型", na=False))]
        if len(fx_rows) > 0:
            self.entry_bi_low_2buy = float(fx_rows.iloc[-1]["fx_b_low"])
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

    def process_bar(self, bar_date, bar_close, bar_high, sell_exit_target=None):
        """Process one bar through exit priority chain (B1: state machine in engine).

        Priority:
          1. Structure stop-loss
          2. V-drop
          3. FULL -> HALF (transition, not exit)
          4. HALF -> buyback / second sell / timeout
          5. Sell signal

        Returns BarResult with new state and exit signal.
        """
        fx_on_bar = self._detect_fx(bar_date)

        if self.state == "EMPTY":
            return BarResult(state="EMPTY", position_pct=0.0, exit=False,
                             trajectory=list(self._trajectory) if self.trajectory_log else None)

        # P1/P2: stateless exit checks
        old_defense = self.defense
        self.update_defense(bar_date)
        if self.defense > old_defense:
            self._log_trajectory("DEFENSE_UP", bar_date, bar_close, detail="止损上移")

        if self.defense > 0 and bar_close <= self.defense:
            self.state = "EMPTY"
            self.position_pct = 0.0
            self._log_trajectory("EXIT", bar_date, bar_close, "结构止损")
            return BarResult(state="EMPTY", position_pct=0.0, exit=True, partial_exits=list(self._partial_exits),
                             exit_reason="结构止损",
                             trajectory=list(self._trajectory) if self.trajectory_log else None)

        if self.check_v_drop(bar_date, bar_close):
            self.state = "EMPTY"
            self.position_pct = 0.0
            self._log_trajectory("EXIT", bar_date, bar_close, "V型暴跌穿GG")
            return BarResult(state="EMPTY", position_pct=0.0, exit=True, partial_exits=list(self._partial_exits),
                             exit_reason="V型暴跌穿GG",
                             trajectory=list(self._trajectory) if self.trajectory_log else None)

        # P3: FULL -> HALF
        if self.state == "FULL":
            half_cut = self.check_half_cut(bar_date, bar_high, fx_on_bar)
            if half_cut:
                self.state = "HALF"
                self.position_pct = 0.5
                self.half_cut_fx_high = half_cut["fx_high"]
                self.half_cut_day_count = 0
                self._high_since_half_cut = 0.0
                self._partial_exits.append((0.5, bar_close))
                self._log_trajectory("HALF_CUT", bar_date, bar_close, "顶分型+背驰+创新高")

        # P4: HALF state
        elif self.state == "HALF":
            self.half_cut_day_count += 1
            self._high_since_half_cut = max(self._high_since_half_cut, bar_high)

            # Buyback (return to skip P5, matches original 'continue')
            if self.check_buyback(bar_high, self.half_cut_fx_high):
                self.state = "FULL"
                self.position_pct = 1.0
                self.rebind_defense(bar_date)
                self.half_cut_day_count = 0
                self._high_since_half_cut = 0.0
                self._partial_exits = []
                self._log_trajectory("BUYBACK", bar_date, bar_close)
                return BarResult(state="FULL", position_pct=1.0, exit=False,
                                 trajectory=list(self._trajectory) if self.trajectory_log else None)

            # Second sell confirmation
            if (self.half_cut_day_count > 1 and
                    self._high_since_half_cut < self.half_cut_fx_high and
                    self.check_second_sell(fx_on_bar)):
                self.state = "EMPTY"
                self.position_pct = 0.0
                self._log_trajectory("EXIT", bar_date, bar_close, "二卖确认")
                return BarResult(state="EMPTY", position_pct=0.0, exit=True, partial_exits=list(self._partial_exits),
                                 exit_reason="二卖确认",
                                 trajectory=list(self._trajectory) if self.trajectory_log else None)

            # Timeout
            if self.half_cut_day_count >= HALF_CUT_TIMEOUT:
                self.state = "EMPTY"
                self.position_pct = 0.0
                self._log_trajectory("EXIT", bar_date, bar_close, "半仓超时")
                return BarResult(state="EMPTY", position_pct=0.0, exit=True, partial_exits=list(self._partial_exits),
                                 exit_reason="半仓超时",
                                 trajectory=list(self._trajectory) if self.trajectory_log else None)

        # P5: Sell signal
        if sell_exit_target is not None and bar_date >= sell_exit_target:
            self.state = "EMPTY"
            self.position_pct = 0.0
            self._log_trajectory("EXIT", bar_date, bar_close, "卖点")
            return BarResult(state="EMPTY", position_pct=0.0, exit=True, partial_exits=list(self._partial_exits),
                             exit_reason="卖点",
                             trajectory=list(self._trajectory) if self.trajectory_log else None)

        return BarResult(state=self.state, position_pct=self.position_pct, exit=False,
                         trajectory=list(self._trajectory) if self.trajectory_log else None)

    def compute_weighted_return(self, entry_price, exit_price):
        """Calculate weighted return accounting for partial exits.

        Uses self._partial_exits = [(pct, price), ...].

        Returns total weighted gross return (before commission).
        """
        if not self._partial_exits:
            return (exit_price - entry_price) / entry_price
        remaining = 1.0
        total = 0.0
        for pct, price in self._partial_exits:
            total += pct * (price - entry_price) / entry_price
            remaining -= pct
        total += remaining * (exit_price - entry_price) / entry_price
        return total

    def _detect_fx(self, bar_date):
        """Detect fractal signal on current bar (moved from runner)."""
        if self.struct_df is None or self.struct_df.empty:
            return None
        bar_d = pd.Timestamp(bar_date).date()
        fx_rows = self.struct_df[
            (pd.to_datetime(self.struct_df["edt"]).dt.date == bar_d) &
            (self.struct_df["fx_b_mark"].str.contains("分型", na=False))
        ]
        if len(fx_rows) == 0:
            return None
        frow = fx_rows.iloc[-1]
        return {
            "mark": str(frow["fx_b_mark"]),
            "high": float(frow.get("fx_b_high", frow.get("high", 0))),
            "low": float(frow.get("fx_b_low", frow.get("low", 0))),
        }

    def update_defense(self, bar_date):
        if self.struct_df is None or self.struct_df.empty:
            return
        bar_d = pd.Timestamp(bar_date).date()
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
        bar_d = pd.Timestamp(bar_date).date()

        if self.struct_30m is not None:
            result = self._check_divergence_30m(bar_d)
            if result is not None:
                return result

        if self.struct_df is not None and not self.struct_df.empty:
            return self._check_divergence_from_struct(bar_d)

        return False

    def _check_divergence_30m(self, bar_date):
        try:
            up_bis = self.struct_30m[
                (self.struct_30m["direction"].str.contains("向上", na=False)) &
                (pd.to_datetime(self.struct_30m["edt"]).dt.date <= bar_date)
            ].sort_values("edt")
            # Tier 1: 优先用非中枢笔（排除震荡噪声）
            up_no_zs = up_bis[up_bis["fx_b_has_zs"] == False]
            if len(up_no_zs) >= 2:
                return float(up_no_zs.iloc[-1]["power"]) < float(up_no_zs.iloc[-2]["power"])
            # Tier 2: 回退到全部向上笔
            if len(up_bis) >= 2:
                return float(up_bis.iloc[-1]["power"]) < float(up_bis.iloc[-2]["power"])
        except Exception:
            pass
        return None

    def _check_divergence_from_struct(self, bar_date):
        sdf = self.struct_df
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
        bar_d = pd.Timestamp(bar_date).date()
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

    def check_buyback(self, bar_high, half_cut_fx_high):
        return bar_high > half_cut_fx_high

    def check_second_sell(self, fx_on_bar):
        if fx_on_bar is None:
            return False
        if "顶分型" not in str(fx_on_bar.get("mark", "")):
            return False
        return True

    def check_v_drop(self, bar_date, bar_close):
        if self.struct_df is None or self.struct_df.empty:
            return False
        sdf = self.struct_df
        bar_d = pd.Timestamp(bar_date).date()
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
        bar_d = pd.Timestamp(bar_date).date()
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
