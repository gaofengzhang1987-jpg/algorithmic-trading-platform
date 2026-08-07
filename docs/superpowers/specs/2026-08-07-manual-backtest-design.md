<!--
  设计文档：人工回测模块 (Manual Backtest)
  日期：2026-08-07
  状态：待审阅
-->

# 人工回测模块设计

## 1. 目标

不改变现有项目逻辑，新增一个人工回测功能模块。用户在 2020-2026 年任意时间截面上输出 L4 候选报告，人工选取股票，按对应信号节点接入出场引擎回测，统计核心指标并与自动 top-N 对比。

## 2. 模块结构

```
manual_backtest/
├── __init__.py
├── engine.py       # ManualBacktester 类（编排层）
├── analyzer.py     # 统计分析 + 人工 vs 自动对比
└── report.py       # CSV/Excel 输出 + 控制台摘要
```

### 2.1 依赖关系

- 仅调用现有模块的公开接口：`EntryFilter`、`L3Filter`、`L4Ranker`、`ExitEngine`、`detect_all_changes`、`load_daily`、`load_signals`、`load_structure_for_code`、`get_next_trading_day`、`get_price_at_date`
- 不修改被调用模块的逻辑
- 唯一改动：`ExitEngine.__init__` 加可选 `trajectory_log` 参数（见 4.3 节）

### 2.2 文件产出

统一放在 `tmp_out/manual_backtest/`，按日期分子目录 `{YYYY-MM-DD}/`。

## 3. ManualBacktester 接口

```python
class ManualBacktester:
    """
    config: dict, 可覆盖以下默认值:
      - sector_rps_min / w_l2 / w_stock_rps / w_sector_rps / w_qlib (L4Ranker 权重)
      - freshness_days: int = 99999  (L3 信号新鲜度, 人工回测默认不限)
    """

    def __init__(self, config: dict | None = None):
        ...

    def run_pipeline(self, date: str) -> pd.DataFrame:
        """对指定截面日期运行 L1→L2→L3→L4，返回 l4_df"""

    def export_for_marking(self, out_dir: str | None = None) -> Path:
        """导出 L4 报告 CSV，含 selected=0 空列"""

    def load_marked(self, path: str | Path) -> pd.DataFrame:
        """读取已标记 CSV (selected=1 的行)"""

    def backtest_selected(self) -> pd.DataFrame:
        """对标记股票逐只回测，返回 trades_df"""

    def backtest_auto_top_n(self, top_n: int = 50) -> pd.DataFrame:
        """自动 top-N 回测（用于对比基准）"""

    # 统计分析由 ManualAnalyzer 独立处理
    #   ManualAnalyzer(trades_df, auto_trades_df).analyze() -> dict
    #   ManualAnalyzer(trades_df, auto_trades_df).compare(top_n) -> pd.DataFrame
```

### 3.1 run_pipeline 流程

```
1. 加载全量信号文件，detect_all_changes 提取当日所有 buy 事件
2. 逐只：load_daily + load_signals + regime 检测
3. EntryFilter.filter → passed + buy_type + total_score
4. 临时覆写 l3_filter.THRESHOLDS[*]["freshness_days"] 为 config.freshness_days（默认 99999），
   try/finally 确保调用后恢复原值 → L3Filter.filter 质量过滤
5. L4Ranker.rank → 排序 + composite + zone_rank
6. 附加 name/sector 字段（从 industry_classification.parquet）
7. 返回 l4_df
```

复用 `smoke_test.py` 中的 `detect_regime()` 和管道逻辑，但不写死 top-N。

### 3.2 backtest_selected 流程

```
对每只标记股票：
  1. 从信号文件中反查 ≤signal_date 且 buy_type 匹配的最近 buy 事件
  2. 以该事件的 next_trading_day 为 entry_date，开盘价为 entry_price
  3. 构造 ExitEngine(code, entry_price, entry_date, buy_type, struct_df, trajectory_log=True)
  4. 逐 bar process_bar，记录出场轨迹
  5. 收集 exit_reason / exit_price / exit_date / return_pct / hold_days / trajectory_json
```

### 3.3 批量模式

外层 Python 循环或 CLI 参数 `--from 2024-01-01 --to 2024-03-31`，每个交易日调用一次 `run_pipeline` + `export_for_marking`。用户标记完后 `load_marked` 支持通配符汇总加载。

## 4. 数据格式

### 4.1 L4 报告 CSV 列

| 列名 | 类型 | 说明 |
|------|------|------|
| selected | int | 0 初始，用户改为 1 |
| global_rank | int | L4 全局排名 |
| zone_rank | int | 买点分区内排名 |
| code | str | 股票代码 |
| name | str | 股票名称 |
| buy_type | str | 一买/二买/三买 |
| signal_date | str | 截面上该信号发生的日期 |
| composite | float | L4 综合评分 |
| n_l2 | float | L2 总分归一化值 |
| stock_rps | float | 个股 RPS |
| sector_rps | float | 板块 RPS |
| qlib_score | float | Qlib ML 预测分数 |
| regime | str | BULL/BEAR/CHOP |
| sector | str | 行业 |
| total_score | float | L2 原始总分 |
| passed | bool | L3 是否通过 |

### 4.2 回测结果 trades_df 列

| 列名 | 类型 | 说明 |
|------|------|------|
| code | str | |
| buy_type | str | |
| signal_date | str | |
| entry_date | str | |
| exit_date | str | |
| entry_price | float | |
| exit_price | float | |
| return_pct | float | 净收益（已扣双向佣金） |
| hold_days | int | |
| exit_reason | str | 结构止损/卖点/到期/半仓超时/V型暴跌/二卖确认 |
| l4_rank | int | |
| composite | float | |
| regime | str | |
| trajectory_json | str | 出场轨迹明细 JSON |

### 4.3 ExitEngine 改动

`ExitEngine.__init__` 新增可选参数 `trajectory_log: bool = False`。
`BarResult`(实际为 @dataclass, 非 namedtuple) 新增 `trajectory: list | None = None` 字段。
`trajectory_log=False` 时 `trajectory=None`, 完全向后兼容。

开启后 `process_bar()` 在每次状态变化时追加记录到 `self._trajectory: list[dict]`：

```python
{
    "date": str,
    "event": str,     # ENTRY / DEFENSE_UP / HALF_CUT / BUYBACK / SECOND_SELL / EXIT
    "price": float,
    "defense": float,
    "state": str,     # FULL / HALF / EMPTY
    "detail": str,    # 可选补充说明
}
```

出场时 `BarResult` 新增可选字段 `trajectory`，默认 `None`。

### 4.4 统计分析输出

`analyze()` 返回 dict：

```python
{
    "summary": {
        "total_trades": int,
        "win_rate": float,
        "avg_return": float,
        "win_loss_ratio": float,
        "avg_hold_days": float,
        "max_win": float,
        "max_loss": float,
        "total_return": float,
    },
    "by_buy_type": {...},    # 按一买/二买/三买分组
    "by_regime": {...},      # 按 BULL/BEAR/CHOP 分组
    "by_exit_reason": {...}, # 按出场原因分组
    "hold_distribution": {...}, # 持仓天数分桶
}
```

`compare_with_auto(n=50)` 返回对比 DataFrame，列：`metric | manual | auto_top_{n}`。

## 5. CLI 入口

```bash
# 单截面
python3 -m manual_backtest date 2024-03-15

# 批量（仅生成报告，不跑回测）
python3 -m manual_backtest batch 2024-01-01 2024-03-31

# 回测已标记
python3 -m manual_backtest backtest tmp_out/manual_backtest/2024-03-15/l4_2024-03-15_marked.csv

# 分析 + 自动对比
python3 -m manual_backtest analyze tmp_out/manual_backtest/2024-03-15/ --compare-top 50
```

## 6. 测试策略

- **单元测试**：`ManualBacktester` 单截面管道（定一只已知股票验证 composite 值与 smoke_test 一致）
- **集成测试**：定一个历史日期，自动 top-3 回测，验证 `backtest_selected()` 与 `compare_with_auto(n=3)` 结果一致
- **回归测试**：运行现有 `smoke_test.py` 确认不改动不影响已有功能
- **ExitEngine trajectory**：单只股票开启 trajectory_log，验证轨迹记录的事件序列与预期一致

## 7. 不做

- 暂不做 Web 界面（后续基于 ManualBacktester 类加 FastAPI 端点）
- 不修改 EntryFilter / L3Filter / L4Ranker 的评分逻辑
- 不引入新的外部依赖
