<!--
  实现计划：人工回测模块
  日期：2026-08-07
  关联设计文档：docs/superpowers/specs/2026-08-07-manual-backtest-design.md
-->

# 人工回测模块 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 manual_backtest/ 模块, 支持任意截面 L4 报告输出, 人工标记, 逐只回测, 统计分析, 与自动 top-N 对比.

**Architecture:** ManualBacktester 类编排 L1-L4 管道和回测流程, ManualAnalyzer 负责统计和对比. ExitEngine 加 trajectory_log 参数输出出场轨迹. 所有输出写入 tmp_out/manual_backtest/.

**Tech Stack:** Python 3, pandas, numpy, 零新增外部依赖.

## Global Constraints

- 不修改 EntryFilter / L3Filter / L4Ranker 的评分逻辑
- 不引入新的外部依赖
- ExitEngine 改动仅限 trajectory_log 参数 (默认 False, 零开销)
- CSV 管道产物统一放在 tmp_out/manual_backtest/{YYYY-MM-DD}/
- 同股同截面 pipeline 输出必须与 smoke_test 一致

---

### Task 1: ExitEngine trajectory_log 参数

**Files:**
- Modify: `backtest/exit_engine.py` (BarResult, __init__, process_bar)

**Interfaces:**
- Produces: `ExitEngine.__init__(..., trajectory_log: bool = False)`, `_trajectory: list[dict]`, `_log_trajectory()`, BarResult 新增 trajectory 字段

- [ ] **Step 1: 修改 BarResult namedtuple, 新增 trajectory 字段**

```python
# Line ~15-20: BarResult 定义
BarResult = namedtuple("BarResult", ["state", "position_pct", "exit", "exit_reason", "partial_exits", "trajectory"])
```

- [ ] **Step 2: __init__ 新增 trajectory_log 参数**

签名末尾加 `trajectory_log: bool = False`, 函数体末尾加:
```python
self.trajectory_log = trajectory_log
self._trajectory: list[dict] = []
```

- [ ] **Step 3: 新增 _log_trajectory 方法**

```python
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
```

- [ ] **Step 4: process_bar 中埋点**

在以下状态变化处调用 _log_trajectory:
1. update_defense 后 defense 上移 → "DEFENSE_UP"
2. 结构止损 → "EXIT" + "结构止损"
3. V-drop → "EXIT" + "V型暴跌穿GG"
4. half_cut → "HALF_CUT" + "顶分型+背驰+创新高"
5. buyback → "BUYBACK"
6. 二卖确认 → "EXIT" + "二卖确认"
7. 半仓超时 → "EXIT" + "半仓超时"
8. 卖点 → "EXIT" + "卖点"
9. 所有 `return BarResult(...)` 改为 `return BarResult(..., trajectory=list(self._trajectory) if self.trajectory_log else None)`

- [ ] **Step 5: 验证默认行为不变**

```bash
python3 -c "from backtest.exit_engine import ExitEngine, BarResult; print('BarResult fields:', BarResult._fields)"
```

- [ ] **Step 6: 提交**

```bash
git add backtest/exit_engine.py
git commit -m "feat: ExitEngine 新增 trajectory_log 可选参数, 记录出场轨迹明细"
```

---

### Task 2: manual_backtest 模块骨架 + report.py

**Files:**
- Create: `manual_backtest/__init__.py`
- Create: `manual_backtest/report.py`

- [ ] **Step 1: 创建 __init__.py** — 导出 ManualBacktester, ManualAnalyzer, export_l4_csv, export_trades_csv, print_summary

- [ ] **Step 2: 创建 report.py** — L4_COLUMNS, TRADE_COLUMNS, export_l4_csv(), export_trades_csv(), print_summary()

- [ ] **Step 3: 提交**

```bash
git add manual_backtest/__init__.py manual_backtest/report.py
git commit -m "feat: manual_backtest 模块骨架 + CSV/控制台输出工具"
```

---

### Task 3: ManualBacktester 核心引擎

**Files:**
- Create: `manual_backtest/engine.py`

**Interfaces:**
- `ManualBacktester(config) -> __init__`
- `run_pipeline(date: str) -> pd.DataFrame`
- `export_for_marking(out_dir=None) -> Path`
- `load_marked(path) -> pd.DataFrame`
- `backtest_selected() -> pd.DataFrame`
- `backtest_auto_top_n(top_n=50) -> pd.DataFrame`

- [ ] **Step 1: 创建 engine.py** — 含 detect_regime, _get_industry_map, ManualBacktester 完整实现 (约 250 行)

- [ ] **Step 2: 快速烟雾测试**

```bash
python3 -c "
from manual_backtest.engine import ManualBacktester
bt = ManualBacktester()
l4 = bt.run_pipeline('2024-08-02')
print(f'L4 candidates: {len(l4)}')
if len(l4)>0: print(l4[['code','buy_type','composite','global_rank']].head(5).to_string())
"
```

- [ ] **Step 3: 提交**

```bash
git add manual_backtest/engine.py
git commit -m "feat: ManualBacktester 核心引擎 — L1-L4 管道 + 人工标记回测"
```

---

### Task 4: ManualAnalyzer 统计分析

**Files:**
- Create: `manual_backtest/analyzer.py`

**Interfaces:**
- `ManualAnalyzer(trades_df, auto_trades_df=None)`
- `analyze() -> dict`
- `compare(top_n=50) -> pd.DataFrame`

- [ ] **Step 1: 创建 analyzer.py** — _compute_summary, _group_stats, analyze, _hold_distribution, compare (约 120 行)

- [ ] **Step 2: 快速功能测试**

```bash
python3 -c "
from manual_backtest.analyzer import ManualAnalyzer
import pandas as pd
df = pd.DataFrame([{'code':'000001','buy_type':'一买','return_pct':5.0,'hold_days':10,'exit_reason':'卖点','regime':'BULL'},{'code':'000002','buy_type':'二买','return_pct':-3.0,'hold_days':5,'exit_reason':'结构止损','regime':'BEAR'}])
a = ManualAnalyzer(df); s = a.analyze()
print(f'total={s[\"summary\"][\"total_trades\"]}, win_rate={s[\"summary\"][\"win_rate\"]}')
"
```

- [ ] **Step 3: 提交**

```bash
git add manual_backtest/analyzer.py
git commit -m "feat: ManualAnalyzer — 分层统计 + 人工 vs 自动对比"
```

---

### Task 5: CLI 入口 __main__.py

**Files:**
- Create: `manual_backtest/__main__.py`

- [ ] **Step 1: 创建 __main__.py** — argparse 四个子命令: date, batch, backtest, analyze

- [ ] **Step 2: 验证 CLI**

```bash
python3 -m manual_backtest --help
python3 -m manual_backtest date 2024-08-02
```

- [ ] **Step 3: 提交**

```bash
git add manual_backtest/__main__.py
git commit -m "feat: manual_backtest CLI 入口 — date/batch/backtest/analyze"
```

---

### Task 6: 集成测试

**Files:**
- Create: `tests/test_manual_backtest.py`

- [ ] **Step 1: 创建测试文件** — 4 个测试: run_pipeline, export_roundtrip, backtest_selected, analyzer

- [ ] **Step 2: 运行测试**

```bash
python3 -m pytest tests/test_manual_backtest.py -v
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_manual_backtest.py
git commit -m "test: ManualBacktester + ManualAnalyzer 集成测试"
```

---

### Task 7: 回归验证

- [ ] **Step 1: 运行 smoke_test**

```bash
python3 backtest/smoke_test.py 2024-02-05 --top 3
```

- [ ] **Step 2: 验证 BarResult 向后兼容**

```bash
python3 -c "from backtest.exit_engine import ExitEngine, BarResult; r = BarResult(state='FULL', position_pct=1.0, exit=False, exit_reason='', partial_exits=[], trajectory=None); print(f'trajectory={r.trajectory}')"
```

- [ ] **Step 3: 确认 git status 干净无意外改动**

```bash
git status
```
