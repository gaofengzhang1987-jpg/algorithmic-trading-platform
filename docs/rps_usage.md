# RPS 数据使用指南

## 文件清单

| 文件 | 格式 | 大小 | 说明 |
|------|------|------|------|
| `data/reference/close_matrix.parquet` | wide (date × code) | 25 MB | 收盘价矩阵，pct_change 依赖的源数据 |
| `data/reference/stock_rps.parquet` | long (一行为一只股票一天) | 162 MB | 个股 RPS，含 ret 和 rps 字段 |
| `data/reference/industry_rps.parquet` | long (一行为一个行业一天) | ~1 MB | 板块 RPS，中位数聚合 |
| `data/daily_close.parquet` | long (date, code, close) | 25 MB | 合并收盘价文件，增量刷新的数据源 |

## 字段定义

### 个股 RPS (stock_rps.parquet)

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | datetime | 交易日 |
| `code` | str | 6 位股票代码 |
| `name` | str | 股票名称 |
| `ret_20d` / `60d` / `120d` / `250d` | float32 | 各周期收益率（%） |
| `rps_20d` / `60d` / `120d` / `250d` | Int64 | 全市场百分位排名（1-99），`<NA>` 代表上市未满周期 |
| `rps_composite` | Int64 | 合成 RPS = rps_120×0.5 + rps_250×0.3 + rps_60×0.2 |
| `pool_size` | int | 当日参与排名的股票总数 |

### 板块 RPS (industry_rps.parquet)

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | datetime | 交易日 |
| `industry` | str | 申万行业名称 |
| `stock_count` | int | 该行业当日有数据的股票数 |
| `rps_20d` / `60d` / `120d` / `250d` | float64 | 行业内个股 RPS 的中位数 |
| `rps_composite` | float64 | 行业内个股 composite RPS 的中位数 |
| `top_10_pct_count` | int | 该行业 rps_composite ≥ 90 的股票数 |

## 常用查询

```python
import pandas as pd

stock = pd.read_parquet('data/reference/stock_rps.parquet')
industry = pd.read_parquet('data/reference/industry_rps.parquet')

# 最新日，RPS 合成排名前 20 的股票
latest = stock[stock['date'] == stock['date'].max()]
latest.nlargest(20, 'rps_composite')

# 某只股票的全部 RPS 历史
stock[stock['code'] == '000001'].set_index('date')[['rps_120d', 'rps_250d', 'rps_composite']]

# 今天最强板块 TOP 10
latest_ind = industry[industry['date'] == industry['date'].max()]
latest_ind.nlargest(10, 'rps_composite')[['industry', 'stock_count', 'rps_composite', 'top_10_pct_count']]

# 找出某个板块内 RPS 排名靠前的股票（需要 join industry）
ind_map = pd.read_parquet('data/industry_classification.parquet')[['code', 'industry']]
merged = latest.merge(ind_map, on='code')
merged[merged['industry'] == '半导体'].nlargest(10, 'rps_composite')
```

## 更新流程

### 全量初始化（仅需一次）
```bash
python3 rps_calc.py full    # ~3 min
```

### 每日增量更新
1. 先确保日线数据已拉取：`python3 data_fetcher.py`
2. 执行 RPS 增量更新：
```bash
python3 rps_calc.py refresh  # ~5-10 秒（有 daily_close.parquet）
```

> `data_fetcher.py` 拉取完成后会自动同步 `daily_close.parquet`，供 refresh 快速读取。
> 如果 `daily_close.parquet` 不存在，refresh 会回退到扫描 4991 文件（~50 秒），并自动生成 `daily_close.parquet`。

## 性能对比

| 操作 | 优化前 | 优化后 |
|------|--------|--------|
| refresh（有 `daily_close`） | ~120s（扫描 4991 文件 + 全量重算） | **~5-10s**（读 1 文件 + 增量计算） |
| refresh（无 `daily_close`，fallback） | 同上 | ~50s（扫描 4991 文件，仅首次触发） |
| full 初始化 | ~150s | 不变（读取 4991 文件，必须） |

> 2026-07-21 优化：瓶颈从 4991 个 parquet 随机读 → 1 个合并文件直读，增量刷新时间缩短 **10-20x**。
