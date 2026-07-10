#!/usr/bin/env python3
"""缠论信号回测引擎 — 基于 Parquet 信号文件的状态变化模拟交易。

回测规则：
  - 入场：买点信号次交易日开盘价买入
  - 出场：卖点信号次交易日开盘价卖出 / 持仓超 N 天 / 止损触发
  - 佣金：0.1%（印花税 + 手续费）
  - 单次买入仓位：固定金额（默认10万元）

注意事项：
  - 笔计算使用了全局数据，存在轻微前视偏差
  - 回测结果仅供参考，不代表实盘收益
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from czsc import CZSC, RawBar, Freq
from czsc.objects import Direction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "daily"
SIGNALS_DIR = BASE_DIR / "data" / "signals"

COMMISSION = 0.001
CAPITAL_PER_TRADE = 100_000
MAX_HOLD_DAYS = 99999
STOP_LOSS_PCT = -0.08

# -- Entry stop discounts (vuln 8/14/31) --
DISCOUNT_BUY1 = 0.96
DISCOUNT_BUY2 = 0.92
DISCOUNT_BUY3 = 0.95

# -- Half-cut params --
HALF_CUT_TIMEOUT = 30

# 买卖点信号列
BUY1_COL = "日线_D1B_BUY1"
BS2_COL  = "日线_D1#SMA#21_BS2辅助V230320"
BS3_COL  = "日线_D1#SMA#34_BS3辅助V230318"
SANMAI_COL = "日线_D1_三买辅助V230228"
SANMAI2_COL = "日线_D1#SMA#34_BS3辅助V230319"
SELL1_COL  = "日线_D1B_SELL1"
# 注意: 二卖/三卖信号列也需检测
SELL_COLS = [SELL1_COL, "日线_D1#SMA#21_BS2辅助V230320", "日线_D1#SMA#34_BS3辅助V230318"]


def _parse_signal(val):
    if pd.isna(val) or str(val) == "0":
        return {"v1": "", "v2": "", "v3": "", "score": "0"}
    parts = str(val).rsplit("_", 3)
    return {"v1": parts[0] if len(parts) >= 4 else "",
            "v2": parts[1] if len(parts) >= 4 else "",
            "v3": parts[2] if len(parts) >= 4 else "",
            "score": parts[3] if len(parts) >= 4 else "0"}


def _detect_all_changes(df) -> list[dict]:
    """检测全部历史中的买点/卖点信号变化。

    Returns:
        [{idx, date, type: "buy"|"sell", signal_label, change_label}, ...]
        按时间排序
    """
    buy_cols = [c for c in [BUY1_COL, BS2_COL, BS3_COL, SANMAI_COL, SANMAI2_COL] if c in df.columns]
    sell_cols = [c for c in SELL_COLS if c in df.columns]
    changes = []

    # 检测买点变化
    for col in buy_cols:
        for i in range(1, len(df)):
            old_r = str(df.iloc[i - 1][col])
            new_r = str(df.iloc[i][col])
            if old_r == new_r:
                continue
            new_p = _parse_signal(new_r)
            if any(k in new_p["v1"] for k in ["一买", "二买", "三买"]):
                dt = df.iloc[i]["dt"]
                sig_label = f"{new_p['v1']}({col.split('_')[2][:8] if '_' in col else col[:8]})"
                changes.append({
                    "idx": i, "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                    "type": "buy", "signal_label": sig_label,
                })

    # 检测卖点变化
    for col in sell_cols:
        for i in range(1, len(df)):
            old_r = str(df.iloc[i - 1][col])
            new_r = str(df.iloc[i][col])
            if old_r == new_r:
                continue
            new_p = _parse_signal(new_r)
            if any(k in new_p["v1"] for k in ["一卖", "二卖", "三卖"]):
                dt = df.iloc[i]["dt"]
                sig_label = new_p["v1"]
                changes.append({
                    "idx": i, "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                    "type": "sell", "signal_label": sig_label,
                })

    changes.sort(key=lambda x: x["date"])
    return changes


def load_daily(code: str) -> pd.DataFrame | None:
    p = DATA_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def get_next_trading_day(date_str: str, daily: pd.DataFrame) -> Optional[pd.Timestamp]:
    sig_date = pd.Timestamp(date_str)
    mask = daily["date"] > sig_date
    future = daily.loc[mask, "date"]
    if future.empty:
        return None
    return future.iloc[0]


def get_price_at_date(date_ts: pd.Timestamp, daily: pd.DataFrame) -> Optional[float]:
    row = daily[daily["date"] == date_ts]
    if row.empty:
        return None
    return float(row["open"].iloc[0])




# 结构缓存 (预计算的 BI低点 + GG高点)
_structure_cache = None

# 模块级加载 (避免重复 IO)
_struct_path = BASE_DIR / "data" / "reference" / "structure_cache.parquet"
_structure_cache = pd.read_parquet(_struct_path) if _struct_path.exists() else None

def _load_structure_cache():
    return _structure_cache

def _lookup_structure(code, target_date_str):
    """查缓存中 <= target_date 的最近结构值。返回 (bi_low, gg_high) 或 (0,0)"""
    sc = _load_structure_cache()
    if sc is None:
        return 0, 0
    target_dt = pd.Timestamp(target_date_str).date()
    mask = (sc["code"] == code) & (sc["dt"] <= target_dt)
    subset = sc[mask]
    if len(subset) == 0:
        return 0, 0
    bi = subset[subset["bi_low"] > 0]
    gg = subset[subset["gg_high"] > 0]
    bi_low = float(bi["bi_low"].iloc[-1]) if len(bi) > 0 else 0
    gg_high = float(gg["gg_high"].iloc[-1]) if len(gg) > 0 else 0
    return bi_low, gg_high

_czsc_date_cache = {}  # (code, date_str) -> CZSC or None

def _get_czsc_at_date(code, target_date_str):
    """获取截至 target_date 的 CZSC 对象（用 ≤ 该日期的日线数据重建）。"""
    key = (code, target_date_str)
    if key in _czsc_date_cache:
        return _czsc_date_cache[key]
    
    dp = DATA_DIR / f"{code}.parquet"
    if not dp.exists():
        _czsc_date_cache[key] = None
        return None
    
    df = pd.read_parquet(dp)
    target_dt = pd.Timestamp(target_date_str).date()
    cutoff = df[df['date'] <= target_dt]
    if len(cutoff) < 500:  # INIT_N 不足, 跳过
        _czsc_date_cache[key] = None
        return None
    
    bars = []
    for i, (_, row) in enumerate(cutoff.iterrows()):
        bars.append(RawBar(
            symbol=code, id=i + 1,
            dt=row["date"].to_pydatetime(), freq=Freq.D,
            open=row["open"], close=row["close"],
            high=row["high"], low=row["low"],
            vol=row.get("volume", 0), amount=row.get("amount", 0),
        ))
    
    try:
        c = CZSC(bars)
    except Exception:
        _czsc_date_cache[key] = None
        return None
    
    _czsc_date_cache[key] = c
    return c


def _get_structure_stop(code, buy_type, entry_date_str, entry_price):
    """从结构缓存查询止损价。"""
    bi_low, gg_high = _lookup_structure(code, entry_date_str)
    if bi_low <= 0:
        return 0  # 缓存无数据
    
    if "一买" in str(buy_type):
        return bi_low
    elif "二买" in str(buy_type):
        # 一买低点 = 上一个向下笔的低点, 但简单取 bi_low
        return bi_low
    elif "三买" in str(buy_type):
        return max(bi_low, gg_high) if gg_high > 0 else bi_low
    return bi_low

# 旧版本 (CZSC-based, 已废弃)
def _get_structure_stop_old(code, buy_type, entry_date_str, entry_price):
    """根据买点类型获取结构性止损价。
    Returns: stop_price or 0 (无法计算)
    
    一买: 底分型最低点
    二买: max(一买低点, 二买底分型最低点)
    三买: max(GG=中枢最高点, 三买底分型最低点)
    """
    c = _get_czsc_at_date(code, entry_date_str)
    if c is None or not c.bi_list:
        return 0  # 数据不足, 返回0表示无法计算结构止损
    
    # 获取所有向下笔的低点 (作为支撑参考)
    down_lows = []
    for bi in c.bi_list:
        if bi.direction == Direction.Down:
            down_lows.append(float(bi.low))
    
    if not down_lows:
        return 0
    
    # 获取底分型数据 (如果有)
    fx_lows = []
    if hasattr(c, 'fx_list') and c.fx_list:
        for fx in c.fx_list:
            if hasattr(fx, 'low') and fx.low > 0:
                fx_lows.append(float(fx.low))
    
    # 一买: 最后一个底分型的最低点
    if "一买" in str(buy_type):
        if fx_lows:
            return fx_lows[-1]
        return down_lows[-1]  # fallback: 最后一个向下笔低点
    
    # 二买: max(一买低点, 二买底分型)
    if "二买" in str(buy_type):
        yi_low = down_lows[-1] if down_lows else 0
        fx_low = fx_lows[-1] if fx_lows else 0
        if yi_low > 0 and fx_low > 0:
            return max(yi_low, fx_low)
        return yi_low or fx_low
    
    # 三买: max(GG, 底分型)
    if "三买" in str(buy_type):
        # GG ≈ 最近中枢最高点, 用向上笔的高点近似
        up_highs = []
        for bi in c.bi_list:
            if bi.direction != Direction.Down:
                up_highs.append(float(bi.high))
        gg = up_highs[-1] if up_highs else 0
        fx_low = fx_lows[-1] if fx_lows else 0
        if gg > 0 and fx_low > 0:
            return max(gg, fx_low)
        return gg or fx_low
    
    # fallback
    return down_lows[-1] if down_lows else 0

# ==================================================================
#  ExitEngine: chan theory structural exit strategy
# ==================================================================

REFERENCE_DIR = BASE_DIR / "data" / "reference"
MIN30_DIR = BASE_DIR / "data" / "min30"


def _load_structure_cache(code):
    p = REFERENCE_DIR / f"{code}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def _load_min30_bars(code):
    p = MIN30_DIR / f"{code}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


class ExitEngine:
    # chan theory structural exit strategy engine
    # Internal state: entry_price, entry_date, buy_type, init_stop, defense
    # defense only moves UP, never down (vuln 7)

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


def simulate_trades(code: str, max_trades: int = 15) -> list[dict]:
    """Chan theory structural exit strategy with position state machine.

    Exit priority (per bar, earliest wins):
      1. Structure stop: close <= defense -> full exit
      2. V-drop: close < last up-pivot GG -> full exit
      3. FULL -> half-cut (top-fx + divergence + new high)
      4. HALF -> buyback / second-sell confirm / 30-day timeout
      5. Sell signal: next trading day open
    """
    daily = load_daily(code)
    if daily is None:
        return []

    sig_path = SIGNALS_DIR / f"{code}.parquet"
    if not sig_path.exists():
        return []
    sig_df = pd.read_parquet(sig_path)

    changes = _detect_all_changes(sig_df)
    if not changes:
        return []

    buy_events = [c for c in changes if c["type"] == "buy"]
    sell_events = [c for c in changes if c["type"] == "sell"]
    if not buy_events:
        return []

    struct_df = _load_structure_cache(code)
    daily_sorted = daily.sort_values("date").reset_index(drop=True)
    trades = []

    for buy in buy_events:
        entry_date = get_next_trading_day(buy["date"], daily_sorted)
        if entry_date is None:
            continue
        entry_price = get_price_at_date(entry_date, daily_sorted)
        if entry_price is None or entry_price <= 0:
            continue

        engine = ExitEngine(code, entry_price, entry_date, buy["signal_label"], struct_df)

        # Pre-calculate first sell signal target exit date (vuln 32 fix)
        sell_exit_target = None
        sell_exit_label = ""
        for sell in sell_events:
            sell_date = pd.Timestamp(sell["date"])
            if sell_date > entry_date:
                cand = get_next_trading_day(sell["date"], daily_sorted)
                if cand is not None:
                    sell_exit_target = cand
                    sell_exit_label = sell["signal_label"]
                break

        # Position state machine
        state = "FULL"
        half_cut_fx_high = 0
        half_cut_day_count = 0
        position_pct = 1.0

        exit_date = None
        exit_price = None
        exit_reason = ""

        entry_idx = daily_sorted[daily_sorted["date"] == entry_date].index
        if entry_idx.empty:
            continue
        entry_idx = entry_idx[0]

        end_idx = min(entry_idx + MAX_HOLD_DAYS + 1, len(daily_sorted))
        window = daily_sorted.iloc[entry_idx + 1 : end_idx]
        if len(window) == 0:
            continue

        earliest_exit = None

        for bi in range(len(window)):
            bar = window.iloc[bi]
            bar_date = bar["date"]
            bar_close = bar["close"]
            bar_high = bar["high"]

            # FX on this bar
            fx_on_bar = None
            if struct_df is not None and not struct_df.empty:
                bar_d = bar_date.date() if hasattr(bar_date, 'date') else bar_date
                if isinstance(bar_d, pd.Timestamp):
                    bar_d = bar_d.date()
                fx_rows = struct_df[
                    (pd.to_datetime(struct_df["edt"]).dt.date == bar_d) &
                    (struct_df["fx_b_mark"].str.contains("\u5206\u578b", na=False))
                ]
                if len(fx_rows) > 0:
                    frow = fx_rows.iloc[-1]
                    fx_on_bar = {
                        "mark": str(frow["fx_b_mark"]),
                        "high": float(frow["fx_b_high"]),
                        "low": float(frow["fx_b_low"]),
                    }

            if state != "EMPTY":
                engine.update_defense(bar_date)

            # 1) Structure stop
            if state != "EMPTY" and engine.defense > 0 and bar_close <= engine.defense:
                earliest_exit = (bi, "\u7ed3\u6784\u6b62\u635f", bar_close, position_pct)
                break

            # 2) V-drop
            if state != "EMPTY" and engine.check_v_drop(bar_date, bar_close):
                earliest_exit = (bi, "V\u578b\u66b4\u8dcc\u7a7fGG", bar_close, position_pct)
                break

            # 3) Position-dependent
            if state == "FULL":
                half_cut = engine.check_half_cut(bar_date, bar_high, fx_on_bar)
                if half_cut:
                    half_cut_fx_high = half_cut["fx_high"]
                    half_cut_day_count = 0
                    position_pct = 0.5
                    state = "HALF"

            elif state == "HALF":
                half_cut_day_count += 1

                if engine.check_buyback(bar_date, bar_high, half_cut_fx_high):
                    position_pct = 1.0
                    state = "FULL"
                    engine.rebind_defense(bar_date)
                    half_cut_day_count = 0
                    continue

                if half_cut_day_count > 1:
                    start_i = max(0, bi - half_cut_day_count + 1)
                    max_high_since = window["high"].iloc[start_i : bi + 1].max()
                    if max_high_since < half_cut_fx_high:
                        second_sell = engine.check_second_sell(bar_date, bar_high, half_cut_fx_high, fx_on_bar)
                        if second_sell:
                            earliest_exit = (bi, "\u4e8c\u5356\u786e\u8ba4", bar_close, position_pct)
                            break

                if half_cut_day_count >= HALF_CUT_TIMEOUT:
                    earliest_exit = (bi, "\u534a\u4ed3\u8d85\u65f6", bar_close, position_pct)
                    break

            # 4) Sell signal: target pre-calculated before loop
            if sell_exit_target is not None and bar_date >= sell_exit_target:
                sp = get_price_at_date(bar_date, daily_sorted) or bar_close
                earliest_exit = (bi, "\u5356\u70b9", sp, position_pct)
                break

        if earliest_exit is not None:
            exit_idx, exit_reason, exit_price_val, _ = earliest_exit
            exit_date = window.iloc[exit_idx]["date"]
            exit_price = exit_price_val
            exit_reason = exit_reason
        else:
            exit_date = window.iloc[-1]["date"]
            exit_price = float(window.iloc[-1]["open"])
            exit_reason = "\u5230\u671f"

        if exit_price is None and exit_date is not None:
            exit_price = get_price_at_date(exit_date, daily_sorted)
        if exit_price is None or exit_price <= 0:
            continue

        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - 2 * COMMISSION
        hold_days = (exit_date - entry_date).days

        trades.append({
            "code": code,
            "signal_type": buy["signal_label"],
            "signal_date": buy["date"],
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "return_pct": round(net_return * 100, 2),
            "hold_days": hold_days,
            "exit_reason": exit_reason,
        })

    return trades

def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "avg_return": 0, "total_return": 0,
                "max_return": 0, "min_return": 0, "avg_hold_days": 0, "sharpe": 0}

    returns = [t["return_pct"] / 100 for t in trades]
    hold_days = [t["hold_days"] for t in trades]
    wins = sum(1 for r in returns if r > 0)

    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    avg_return = np.mean(returns) if returns else 0
    total_return = sum(returns)
    max_r = max(returns) if returns else 0
    min_r = min(returns) if returns else 0
    avg_hold = np.mean(hold_days) if hold_days else 0

    if len(returns) > 1 and avg_hold > 0:
        daily_std = np.std(returns)
        sharpe = (avg_return / (daily_std + 1e-8)) * np.sqrt(252 / avg_hold)
    else:
        sharpe = 0

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate * 100, 1),
        "avg_return": round(avg_return * 100, 2),
        "total_return": round(total_return * 100, 2),
        "max_return": round(max_r * 100, 2),
        "min_return": round(min_r * 100, 2),
        "avg_hold_days": round(avg_hold, 1),
        "sharpe": round(sharpe, 2),
    }


def run_all(codes: list[str] | None = None) -> pd.DataFrame:
    if codes is None:
        codes = sorted(p.stem for p in SIGNALS_DIR.glob("*.parquet"))

    all_trades = []
    for i, code in enumerate(codes):
        try:
            trades = simulate_trades(code)
            all_trades.extend(trades)
        except Exception as e:
            logger.warning("%s: 回测失败: %s", code, str(e)[:80])
        if (i + 1) % 200 == 0:
            logger.info("回测进度: %d/%d, 累计交易 %d 笔", i + 1, len(codes), len(all_trades))

    df = pd.DataFrame(all_trades)
    if df.empty:
        logger.info("回测结果为空")
        return df

    df.sort_values("entry_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("回测完成: %d 只股票, %d 笔交易", len(set(df["code"])), len(df))
    return df


def main():
    import sys
    codes_arg = None
    if len(sys.argv) > 1:
        codes_arg = sys.argv[1:]
    df = run_all(codes_arg)
    if df.empty:
        print("无交易记录")
        return
    metrics = compute_metrics(df.to_dict("records"))
    print("=== 回测绩效 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n前 10 笔交易:")
    print(df.head(10)[["code", "signal_type", "entry_date", "exit_date", "return_pct", "exit_reason"]].to_string())


if __name__ == "__main__":
    main()
