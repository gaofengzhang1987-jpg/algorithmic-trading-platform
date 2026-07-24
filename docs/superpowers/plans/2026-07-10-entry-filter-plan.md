# EntryFilter Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task.

**Goal:** Insert `EntryFilter` before `simulate_trades` entry loop to filter low-quality buy signals.

**Architecture:** New `EntryFilter` class in `backtest.py` with P0 hard gates + P1 weighted scoring. Reuses existing scoring functions from `zone2_pattern.py` where possible. CZSC object created once per stock in `simulate_trades`, shared across all buy events for that stock.

**Tech Stack:** Python, czsc, pandas, numpy

## Global Constraints

- Do not modify `ExitEngine`, `precompute_structure.py`, or `strategy.py`
- Reuse `_score_stroke`, `_score_volume`, `_score_macd` from `zone2_pattern.py`
- CZSC created once per stock, not per buy event
- One buy type per filter call; 一买+二买同bar → score both, take max

---

### Task 1: Extract scoring functions from zone2_pattern.py

**Files:**
- Read: `zone2_pattern.py` (all)
- No modify — copy functions into EntryFilter

**Produce:** Scoping report — which functions to import vs rewrite inline

**Check:**
- [ ] `_score_stroke(b1v, smv)` — clean importable, takes 2 signal values → 0-100
- [ ] `_score_volume(df)` — clean importable, takes daily.tail(VOL_MA) → 0-100
- [ ] `_score_macd(df)` — clean importable, takes signal.tail(MACD_WIN) → 0-100
- [ ] `_score_divergence(code, cp)` — uses global `_czsc_cache`, returns tuple(div, stroke, fractal, pivot). Must rewrite into separate methods in EntryFilter
- [ ] `_is_bottom_fractal(fx)` — clean importable, detects bottom fx

Verify: `python3 -c "from zone2_pattern import _score_stroke, _score_volume, _score_macd; print('OK')"`

---

### Task 2: Create EntryFilter class skeleton

**Files:**
- Modify: `backtest.py` — insert before `simulate_trades`

**Interface:**
```python
class EntryFilter:
    THRESHOLDS = {"一买": 350, "二买": 330, "三买": 280}
    MAX_SCORES = {"一买": 600, "二买": 550, "三买": 400}

    def __init__(self, code, struct_df, daily_df, sig_df, czsc_obj):
        ...

    def check(self, buy_event) -> FilterResult:
        ...
```

**FilterResult:**
```python
class FilterResult:
    passed: bool
    buy_type: str
    total_score: int
    dimension_scores: dict[str, int]
    reject_reason: str
```

Verify: `python3 -m py_compile backtest.py`

---

### Task 3: Implement P0 hard gates

**Files:**
- Modify: `backtest.py` `EntryFilter._check_p0()`

**Logic:**
```python
def _check_p0(self, buy_event):
    signal = buy_event["signal_label"]
    # L2-02: valid buy type
    if not any(t in signal for t in ["一买", "二买", "三买"]):
        return False, "非买卖信号"
    # L2-03: freshness (within 30 days)
    signal_date = pd.Timestamp(buy_event["date"])
    if signal_date < self.latest_bar_date - pd.Timedelta(days=30):
        return False, "信号过期(>30天)"
    return True, None
```

Verify: 3-stock test — count rejected by P0

---

### Task 4: Implement P1 scoring — importable dimensions

**Files:**
- Modify: `backtest.py` `EntryFilter._score_dimensions()`

**Logic:**
```python
def _score_dimensions(self, buy_event, buy_type):
    scores = {}
    signal_val = buy_event["signal_label"]
    signal_date = buy_event["date"]

    scores["笔数"] = _score_stroke(signal_val, "")
    scores["量比"] = _score_volume(self.daily_df)
    scores["MACD"] = _score_macd(self.sig_df)
    # core verification, fractal, pivot → Task 5
    return scores
```

Verify: 3-stock test — print dimension scores for first 5 buy events

---

### Task 5: Implement core verification (背驰/回踩) + fractal + pivot

**Files:**
- Modify: `backtest.py` `EntryFilter._score_core()`, `_score_fractal()`, `_score_pivot()`

**Backtest (一买):**
```python
def _score_divergence_power(self):
    dbs = [bi for bi in self.czsc.bi_list if bi.direction == Direction.Down]
    # Skip pivot-internal BIs
    dbs_no_zs = [bi for bi in dbs if not bi.fx_b.has_zs]
    if len(dbs_no_zs) >= 2:
        a, b = dbs_no_zs[-2], dbs_no_zs[-1]
    elif len(dbs) >= 2:
        a, b = dbs[-2], dbs[-1]
    else:
        return 0
    if a.power <= 0:
        return 0
    ratio = b.power / a.power
    if ratio < 0.3: return 100
    elif ratio < 0.6: return 75
    elif ratio < 0.9: return 40
    elif ratio < 1.0: return 10
    return 0
```

**回踩深度 (二买):**
```python
def _score_retrace_depth(self):
    downs = [bi for bi in self.czsc.bi_list if bi.direction == Direction.Down]
    if len(downs) < 2:
        return 0
    cur = downs[-1]  # 回踩笔
    prev = downs[-2]  # 一买前下跌笔
    cur_range = cur.high - cur.low
    prev_range = prev.high - prev.low
    if prev_range <= 0:
        return 0
    ratio = cur_range / prev_range
    if ratio < 0.3: return 100
    elif ratio < 0.5: return 75
    elif ratio < 0.7: return 40
    elif ratio < 1.0: return 10
    return 0
```

**底分型 (fractal, 所有买点):** Reuse `_is_bottom_fractal` and existing scoring logic from zone2_pattern.py's `_score_divergence` — extract the sk+sc+sv from that function. sc 分配: 强势 0-60, BI端点 0-20, 笔内位置 0-20.

**中枢下沿 (仅一买):**
```python
def _score_pivot_dist(self, price):
    downs = [bi for bi in self.czsc.bi_list if bi.direction == Direction.Down]
    if not downs:
        return 0
    nearest_low = downs[-1].low
    dist_pct = abs(price - nearest_low) / price * 100
    if dist_pct <= 1: return 100
    elif dist_pct <= 2: return 60
    elif dist_pct <= 5: return 20
    return 0
```

Verify: Run on 000001 — print each dimension score for first 3 buys per type

---

### Task 6: Implement total score + threshold check

**Files:**
- Modify: `backtest.py` `EntryFilter.check()`

**Logic:**
```python
WEIGHTS = {
    "一买": {"核心验证": 1.5, "底分型": 1.3, "MACD": 0.7, "笔数": 1.0, "量比": 1.0, "中枢": 0.5},
    "二买": {"核心验证": 1.5, "底分型": 1.3, "MACD": 0.7, "笔数": 1.0, "量比": 1.0},
    "三买": {                "底分型": 1.3, "MACD": 0.7, "笔数": 1.0, "量比": 1.0},
}

def check(self, buy_event):
    p0_ok, p0_reason = self._check_p0(buy_event)
    if not p0_ok:
        return FilterResult(False, reason=p0_reason)

    buy_type = self._detect_buy_type(buy_event)
    scores = self._score_dimensions(buy_event, buy_type)
    weights = self.WEIGHTS[buy_type]
    total = sum(scores[dim] * weights[dim] for dim in scores if dim in weights)
    threshold = self.THRESHOLDS[buy_type]
    passed = total >= threshold

    return FilterResult(
        passed=passed,
        buy_type=buy_type,
        total_score=total,
        dimension_scores=scores,
        reject_reason=None if passed else f"P1总分{total}/{threshold}({buy_type}门槛)"
    )
```

Multi-buy-type handler:
```python
def _detect_buy_type(self, buy_event):
    signal = buy_event["signal_label"]
    types = []
    if "一买" in signal: types.append("一买")
    if "二买" in signal: types.append("二买")
    if "三买" in signal: types.append("三买")
    if len(types) == 1:
        return types[0]
    # Multiple: score all, take max
    best_type, best_score = None, -1
    for t in types:
        s = self._score_dimensions(buy_event, t)
        total = sum(s[d] * self.WEIGHTS[t][d] for d in s if d in self.WEIGHTS[t])
        if total > best_score:
            best_score, best_type = total, t
    return best_type
```

Verify: All buy events now pass through filter

---

### Task 7: Integrate into simulate_trades

**Files:**
- Modify: `backtest.py` `simulate_trades`

**Change:** Insert 3 lines before ExitEngine:
```python
# Load CZSC once per stock for EntryFilter
czsc_obj = _get_czsc_at_date(code, str(daily_sorted["date"].iloc[-1].date()))
entry_filter = EntryFilter(code, struct_df, daily_sorted, sig_df, czsc_obj)

for buy in buy_events:
    result = entry_filter.check(buy)
    if not result.passed:
        continue  # Skip low-quality signals
    # ... existing ExitEngine logic
```

Verify: `python3 -m py_compile backtest.py`

---

### Task 8: End-to-end test — compare filtered vs unfiltered

**Run:**
```python
from backtest import simulate_trades, compute_metrics
from collections import Counter

# Track both passed and rejected
for code in ['000001', '000002', '600519']:
    trades_before = simulate_trades(code)  # without filter
    trades_after = simulate_trades(code, use_filter=True)  # with filter

    m_before = compute_metrics(trades_before)
    m_after = compute_metrics(trades_after)
    print(f"{code}: {m_before['total_trades']}→{m_after['total_trades']} "
          f"win={m_before['win_rate']}%→{m_after['win_rate']}%")
```

Expected: win rate improves, total trades decreases, filter rejects ~30-50% of signals.
