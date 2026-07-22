# Backtest Validation: Optuna EntryFilter 实现计划

> **For agentic workers:** 使用 superpowers:executing-plans 逐任务实现。步骤用 checkbox (`- [ ]`) 跟踪。

**Goal:** 三组 EntryFilter 配置对比 + 6 组阈值扫描

**Architecture:** 复刻 `backtest_full.py` 的回测循环，外层包裹三套 REGIME_WEIGHTS + REGIME_THRESHOLDS 切换。阈值扫描独立脚本，不跑回测、仅用 label 统计。

**Tech Stack:** Python 3.12, pandas, CZSC, struct_df 缓存

## Global Constraints

- 回测日期范围：2020-01-01 ~ 2026-07-21
- struct_df 缓存路径：`data/struct_cache/`
- 复刻 `backtest_full.py` 中的 ExitEngine 逻辑，不修改出场规则
- 阈值扫描使用 `tmp_out/optuna_bullbear_full.parquet`（已有全量 scores）

---

### Task 1: 验证数据完整性

**Files:**
- Read: `data/struct_cache/`（已有缓存）
- Read: `tmp_out/optuna_bullbear_full.parquet`

- [ ] **Step 1: 检查 struct_cache 覆盖率**

```bash
cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
python3 -c "
from pathlib import Path
cache = list(Path('data/struct_cache').glob('*.parquet'))
print(f'struct_cache: {len(cache)} files')
# spot-check one
import pandas as pd
s = pd.read_parquet(cache[0])
print(f'example: {len(s)} rows, cols={list(s.columns[:5])}')
"
```

Expected: 1500+ files, each with bi + pivot data

- [ ] **Step 2: 验证 optuna_bullbear_full.parquet 可用**

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('tmp_out/optuna_bullbear_full.parquet')
print(f'rows: {len(df):,}')
print(f'cols: {sorted(df.columns.tolist())}')
print(f'regimes: {dict(df.regime.value_counts())}')
print(f'buy_types: {dict(df.buy_type.value_counts())}')
# check old dims present
old = ['核心验证','底分型','MACD','量比','笔数','中枢','距离']
print(f'old dims missing: {df[old].isna().any().any()}')
# check new dims present
new = ['MA防守','MA排列强度','底部放量']
print(f'new dims present: {[c for c in new if c in df.columns]}')
"
```

Expected: 1,042,897 rows, 0% missing old dims, 3 new dims present

---

### Task 2: 阈值扫描脚本

**Files:**
- Create: `tmp_out/scan_thresholds.py`

**Interfaces:**
- Consumes: `tmp_out/optuna_bullbear_full.parquet`（Task 1 验证通过）
- Produces: stdout 表格

- [ ] **Step 1: 写脚本**

```python
#!/usr/bin/env python3
"""阈值扫描：6 组 buy_type × regime，固定 Optuna 权重，遍历阈值点。"""
import pandas as pd, numpy as np

df = pd.read_parquet('tmp_out/optuna_bullbear_full.parquet')

# Optuna 权重（硬编码，与 entry_filter.py 一致）
WEIGHTS = {
    ("一买", "BEAR"): {"中枢": 0.57, "量比": 0.44, "MACD": 0.29, "底分型": 0.13, "核心验证": 0.10, "底部放量": 0.10},
    ("一买", "BULL"): {"量比": 0.70, "中枢": 0.45, "MACD": 0.39, "核心验证": 0.34, "距离": 0.26},
    ("二买", "BEAR"): {"MACD": 1.52, "量比": 0.90, "底分型": 0.44, "核心验证": 0.40},
    ("二买", "BULL"): {"MA防守": 2.96, "MACD": 0.99, "笔数": 0.98, "底分型": 0.08},
    ("三买", "BEAR"): {"笔数": 1.95, "量比": 0.96, "MACD": 0.59, "核心验证": 0.20, "底分型": 0.10},
    ("三买", "BULL"): {"MA排列强度": 1.09, "笔数": 0.90, "MACD": 0.72},
}

SCAN_THRESHOLDS = {
    ("一买", "BEAR"): [60, 65, 70, 75, 80, 87, 95, 101],
    ("一买", "BULL"): [97, 109, 117, 122, 129, 135, 144, 151],
    ("二买", "BEAR"): [164, 168, 178, 182, 187, 191, 196, 201],
    ("二买", "BULL"): [370, 400, 408, 412, 416, 418, 419, 421],
    ("三买", "BEAR"): [59, 65, 68, 70, 75, 83, 86, 90],
    ("三买", "BULL"): [118, 124, 157, 190],
}

for (bt, regime), w in WEIGHTS.items():
    data = df[(df["buy_type"] == bt) & (df["regime"] == regime)].copy()
    true_n = (data["label"] == 1).sum()
    total_n = len(data)
    
    data["total"] = 0.0
    for d, weight in w.items():
        if d in data.columns:
            data["total"] += data[d].fillna(0) * weight
    
    print(f"\n{bt} {regime} (n={total_n:,} true={true_n:,})  weights: {w}")
    print(f"  {'阈值':>6}  {'通过':>8}  {'通过率':>7}  {'Prec':>6}  {'Recall':>7}")
    print("  " + "-" * 45)
    
    for th in SCAN_THRESHOLDS.get((bt, regime), []):
        passed = data[data["total"] >= th]
        if len(passed) == 0: continue
        tp = (passed["label"] == 1).sum()
        prec = tp / len(passed)
        rec = tp / true_n
        print(f"  {th:>6}  {len(passed):>8,}  {len(passed)/total_n:.1%}    {prec:.3f}  {rec:.3f}")
```

- [ ] **Step 2: 运行验证**

```bash
cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
python3 tmp_out/scan_thresholds.py
```

Expected: 6 张表各 4-8 行，Precision/Recall 值与设计阶段一致

---

### Task 3: 三组对比 — 回测框架

**Files:**
- Create: `tmp_out/backtest_compare.py`

**Interfaces:**
- Consumes: `data/struct_cache/`, `data/daily/`, `data/signals/`
- Produces: `tmp_out/backtest_compare_results.json`

- [ ] **Step 1: 定义三套配置**

```python
# ====== 配置定义 ======
OLD_WEIGHTS = {
    "一买": {
        "BEAR": {"核心验证": 1.34, "底分型": 0.43, "MACD": 0.89, "量比": 0.98, "中枢": 0.94, "距离": 0.89},
        "BULL": {"核心验证": 1.21, "底分型": 0.89, "MACD": 0.05, "量比": 0.83, "中枢": 0.94, "距离": 1.87},
    },
    "二买": {
        "BEAR": {"核心验证": 1.68, "量比": 0.00, "MACD": 1.57, "笔数": 1.00, "底分型": 0.78},
        "BULL": {"核心验证": 2.11, "量比": 0.16, "MACD": 1.38, "笔数": 1.28, "底分型": 1.61},
    },
    "三买": {
        "BEAR": {"核心验证": 1.12, "量比": 0.05, "MACD": 1.08, "笔数": 1.68, "底分型": 1.96},
        "BULL": {"核心验证": 1.07, "量比": 0.26, "MACD": 1.99, "笔数": 1.45, "底分型": 0.48},
    },
}
OLD_THRESHOLDS = {"一买": {"BEAR": 255, "BULL": 999}, "二买": {"BEAR": 273, "BULL": 327}, "三买": {"BEAR": 999, "BULL": 366}}

NEW_WEIGHTS = { ... }  # Optuna weights (copy from entry_filter.py REGIME_WEIGHTS, CHOP removed)
NEW_THRESHOLDS_OPEN = {"一买": {"BEAR": 70, "BULL": 144}, "二买": {"BEAR": 164, "BULL": 412}, "三买": {"BEAR": 90, "BULL": 190}}
NEW_THRESHOLDS_CLOSED = {"一买": {"BEAR": 70, "BULL": 999}, "二买": {"BEAR": 164, "BULL": 412}, "三买": {"BEAR": 999, "BULL": 190}}

CONFIGS = [
    ("baseline_A", OLD_WEIGHTS, OLD_THRESHOLDS),
    ("baseline_B", NEW_WEIGHTS, NEW_THRESHOLDS_CLOSED),
    ("experiment", NEW_WEIGHTS, NEW_THRESHOLDS_OPEN),
]
```

- [ ] **Step 2: 复刻 backtest_full.py 的回测循环**

从 `tmp_out/backtest_full.py` 复制以下函数到新脚本：
- `_build_struct_df(code)` — 结构缓存（从 `data/struct_cache/` 读，避免重复计算）
- `simulate_exit(code, entry_price, entry_date, buy_type, struct_df, future_df, regime)` — 出场模拟
- 回测主循环：`for code in stocks: load_daily → load_struct → load_signal → for each signal → EntryFilter → ExitEngine`

关键改动：在外层循环中接受 `weights_dict` 和 `thresholds_dict` 参数，供 `EntryFilter` 使用。

- [ ] **Step 3: 实现 EntryFilter 参数注入**

```python
import entry_filter as ef_mod

def run_backtest_pass(name, weights, thresholds, stocks):
    """一次回测遍历，返回 trades 列表。"""
    # Monkey-patch EntryFilter 用到的权重和阈值
    ef_mod.REGIME_WEIGHTS = weights
    ef_mod.REGIME_THRESHOLDS = thresholds
    
    trades = []
    for code in stocks:
        # ... backtest loop (same as backtest_full.py) ...
        result = EntryFilter(code, daily, sig, regime).filter(buy_event)
        # ...
    return trades
```

- [ ] **Step 4: 运行三趟并汇总结果**

```python
results = {}
for name, weights, thresholds in CONFIGS:
    print(f"Running {name}...", flush=True)
    trades = run_backtest_pass(name, weights, thresholds, stocks)
    results[name] = summarize(trades)

# 输出对比表格
print(f"\n{'指标':<18} {'基线A':>10} {'基线B':>10} {'实验组':>10} {'A→C':>8} {'B→C':>8}")
print("-" * 62)
for metric in ["trades", "win_rate", "avg_return", "profit_loss_ratio", "avg_hold_days"]:
    a = results["baseline_A"][metric]
    b = results["baseline_B"][metric]
    c = results["experiment"][metric]
    d1 = f"{c-a:+.1f}" if isinstance(a, float) else f"{c-a:+d}"
    d2 = f"{c-b:+.1f}" if isinstance(b, float) else f"{c-b:+d}"
    print(f"{metric:<18} {a:>10} {b:>10} {c:>10} {d1:>8} {d2:>8}")
```

---

### Task 4: 运行回归验证

- [ ] **Step 1: 跑阈值扫描**

```bash
python3 tmp_out/scan_thresholds.py
```

验证输出与人工选取时的数据匹配。

- [ ] **Step 2: 跑三组对比**

```bash
python3 tmp_out/backtest_compare.py
```

Expected: 三行对比数据，一买 BULL 和三买 BEAR 从 baseline 到 experiment 交易笔数应增加。

- [ ] **Step 3: 保存结果**

```bash
cat tmp_out/backtest_compare_results.json
```

---

### Task 5: 提交

- [ ] **Step 1: 提交代码**

```bash
git add tmp_out/scan_thresholds.py tmp_out/backtest_compare.py
git commit -m "feat: add EntryFilter threshold scan and 3-config backtest comparison"
```

- [ ] **Step 2: 验证完整性**

```bash
python3 -c "import ast; [ast.parse(open(f'tmp_out/{f}').read()) for f in ['scan_thresholds.py','backtest_compare.py']]"
python3 tmp_out/scan_thresholds.py | head -5
```

Expected: 语法检查通过，阈值扫描有输出
