# L2 因子全维度 IC 分析方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 对 L2 EntryFilter 的 20 个维度在 CHOP regime 下按买点类型分别做 IC 分析，输出每个维度的预测力证据，指导 REGIME_WEIGHTS 调优。

**Architecture:** 用现有 `factor_analysis/` 模块的 `FactorExtractor` 提取 300 只股票的 L2 维度分，分 buy_type 做 Alphalens IC 分析，产出 per-dimension IC/IR/spread 表格，与当前 REGIME_WEIGHTS 交叉对比。

**Tech Stack:** pandas, numpy, alphalens-reloaded, 现有 factor_analysis/ 模块

---

## 背景：L2 路由结构

当前 EntryFilter 对 **3 种买点 × 3 种 regime** 各有独立维度集合，共 20 个去重维度，约 60 个 (buy_type, regime, dimension) 组合。权重通过 REGIME_WEIGHTS 字典硬编码。

### 维度路由矩阵

| 维度 | 一买 | 二买 | 三买 | 备注 |
|------|------|------|------|------|
| MACD | BULL/BEAR/CHOP | BULL/BEAR/CHOP | BULL/BEAR/CHOP | 全覆盖，不同买点**评分函数不同** |
| 底分型 | BEAR/CHOP | BULL/BEAR/CHOP | BEAR/CHOP | 全覆盖 |
| 量比 | BULL/BEAR/CHOP | BEAR/CHOP | BEAR | 一买覆盖最广 |
| 核心验证 | BULL/BEAR/CHOP | BEAR | BEAR | 不同买点含义不同 |
| 中枢 | BULL/BEAR/CHOP | — | — | 仅一买 |
| 距离 | BULL/CHOP | — | — | 仅一买 |
| 相对强度 | CHOP | CHOP | CHOP | CHOP 专属 |
| 时间消化 | CHOP | CHOP | CHOP | CHOP 专属 |
| 笔数 | — | BULL | BULL/BEAR | 二三买 |
| 区间位置 | CHOP | — | — | 仅一买 |
| 波动压缩 | CHOP | — | — | 仅一买 |
| 前低防守距离 | — | CHOP | — | 仅二买 |
| 二次放量启动 | — | CHOP | — | 仅二买 |
| 反弹力度 | — | CHOP | — | 仅二买 |
| MA防守 | — | BULL | — | 仅二买 |
| 回抽深度 | — | — | CHOP | 仅三买 |
| 中枢质量 | — | — | CHOP | 仅三买 |
| ZG站稳天数 | — | — | CHOP | 仅三买 |
| ATR扩张比 | — | — | CHOP | 仅三买 |
| 突破量持续性 | — | — | CHOP | 仅三买 |
| 底部放量 | BEAR | — | — | 仅一买BEAR |
| MA排列强度 | — | — | BULL | 仅三买BULL |

**关键复杂性**：同名维度在不同买点类型中使用**不同的评分函数**。例如 MACD 在一买中调用 `_score_macd`，在二买中可能用不同的逻辑。因此 IC 分析必须**按 (buy_type, dimension) 分拆**，不能跨买点合并。

### 预期样本量（300 只股票，CHOP regime）

| buy_type | 预估信号数 | 维度数 | 每维样本 |
|----------|-----------|--------|----------|
| 一买 | ~1,300 | 10 | ~1,300 |
| 二买 | ~1,150 | 8 | ~1,150 |
| 三买 | ~820 | 9 | ~820 |

**BULL/BEAR regime**：总信号 < 200，每维样本不足 30，跳过独立分析。改为 CHOP 结论外推 + 定性判断。

---

## 分析方案（5 步）

### Step 1: 因子提取

用 `FactorExtractor(regime="CHOP", lookback_days=500)` 跑 300 只股票。

- 产出：MultiIndex (date, asset) DataFrame，列为 buy_type + 所有 L2 维度分
- 同时计算 forward returns (5d, 20d)
- 预估耗时：300 × 7s ≈ 35 min（单进程，含 CZSC 初始化）
- 进度写文件 `tmp_out/l2_ic_progress.txt`

### Step 2: 按 buy_type 拆分 + 数据清洗

对每种 buy_type：
- 只保留该 buy_type 有非零值的维度（例如一买不保留"回抽深度"）
- 剔除有效样本 < 50 的维度
- 用 `_build_prices_for_alphalens()` 构建价格矩阵

### Step 3: Alphalens IC 分析

对每个 (buy_type, dimension) 组合：

```python
factor_data = al.utils.get_clean_factor_and_forward_returns(
    single_factor[dim], prices,
    quantiles=5, periods=(5, 20), max_loss=0.50,
)
ic = al.performance.factor_information_coefficient(factor_data)
mean_ret = al.performance.mean_return_by_quantile(factor_data)
```

产出指标：
- `IC_5D` / `IC_20D`：因子值与 5/20 日前向收益的秩相关系数均值
- `IC_IR_5D`：IC 的信息比率（IC_mean / IC_std）
- `spread_5D_bps`：最高分位 − 最低分位的 5 日收益差（基点）

### Step 4: 与当前权重交叉对比

合并当前 `REGIME_WEIGHTS[bt]["CHOP"]` 到分析结果中，输出三张表：

**`tmp_out/l2_factor_ic_一买.csv`（示例）**

| dimension | n | IC_5D | IC_IR_5D | spread_bps | current_weight | 建议 |
|-----------|----|-------|------|-----------|----------------|------|
| 核心验证 | 1300 | +0.12 | 0.35 | 85 | 0.59 | 维持 |
| 量比 | 1300 | +0.08 | 0.22 | 50 | 1.93 | ⬇ 过高 |
| MACD | 1300 | -0.02 | -0.05 | -10 | 0.07 | 考虑移除 |
| 底分型 | 1300 | +0.15 | 0.42 | 120 | 2.05 | ⬆ 可加 |
| ... |

### Step 5: 权重调整建议

基于 IC 分析结果，输出三组建议：

1. **维度删除候选**：|IC| < 0.02 且 IR < 0.1 → 考虑从 REGIME_WEIGHTS 中移除
2. **权重上调候选**：|IC| > 0.10 且 IR > 0.3 → 增加权重
3. **权重下调候选**：当前权重高但 IC 低 → 降低权重

---

## 执行任务

### Task 1: 写入分析脚本

**Files:**
- Create: `tmp_out/l2_ic_analysis.py`

**目标：** 一体化脚本，从提取 → 拆分 → Alphalens → 输出完整走通。

核心逻辑骨架（伪代码）：

```python
# 1. 提取
ext = FactorExtractor(regime="CHOP", lookback_days=500)
factor_df, _ = ext.extract(max_stocks=300)

# 2. 按 buy_type 拆分
for bt in ["一买", "二买", "三买"]:
    bt_df = factor_df[factor_df["buy_type"] == bt]
    prices = _build_prices_for_alphalens(bt_df)
    
    for dim in valid_dimensions(bt_df):
        # 3. Alphalens
        factor_data = al.utils.get_clean_factor_and_forward_returns(...)
        ic = al.performance.factor_information_coefficient(factor_data)
        
        # 4. 记录结果 + 权重对比
        record = {"buy_type": bt, "dimension": dim, "IC_5D": ...,
                  "current_weight": REGIME_WEIGHTS[bt]["CHOP"].get(dim, 0)}
    
    # 5. 输出 CSV
    df_bt.to_csv(f"tmp_out/l2_factor_ic_{bt}.csv")
```

**验证：** `python3 tmp_out/l2_ic_analysis.py` 跑通，产出 3 张 CSV + 1 张汇总表。

### Task 2: 执行 + 结果解读

**Files:**
- Read: `tmp_out/l2_factor_ic_一买.csv`, `...二买.csv`, `...三买.csv`
- Read: `tmp_out/l2_factor_ic_summary.csv`

**步骤：**
- [ ] 启动后台运行：`python3 -u tmp_out/l2_ic_analysis.py > tmp_out/l2_ic_analysis.log 2>&1 &`
- [ ] 每 5 分钟检查 `tmp_out/l2_ic_progress.txt`
- [ ] 完成后读取三张 CSV，排序出 TOP/BOTTOM 维度
- [ ] 交叉对比当前权重，标注过重/过轻/无效维度

### Task 3: 生成 REGIME_WEIGHTS 调整建议文档

**Files:**
- Create: `docs/L2因子IC审计报告_20260805.md`

**内容：**
- 三张 buy_type 的因子排名表（含 IC + 当前权重 + 建议权重）
- 维度删除/新增建议
- CHOP 外推到 BULL/BEAR 的定性判断
- 最终推荐的 REGIME_WEIGHTS 数值

---

## 风险与边界

| 风险 | 缓解 |
|------|------|
| 300 只股票中某些特定维度样本 < 50 | 跳过该维度，标注"样本不足" |
| BULL/BEAR 样本不足 | 不做定量 IC，用 CHOP 结论外推 |
| 同名维度不同评分函数的比较 | 严格按 buy_type 分拆，不做跨类型合并 |
| CZSC segfault 导致脚本中断 | 用 subprocess.Popen 分 4 核，每核 ~75 只，per-50 存盘 |
| Alphalens max_loss 过高导致维度跳过 | 设 max_loss=0.50 宽容模式 |

---

## 预估耗时

| 步骤 | 耗时 | 说明 |
|------|------|------|
| 因子提取（300 只） | 35 min | 单核 CZSC 初始化 + EntryFilter |
| Alphalens IC 分析 | 10 min | 每维度 1-2s，~27 个有效维度 |
| 结果解读 + 文档 | 10 min | 人工/AI 解读 |
| **总计** | **~55 min** | |

---

## BULL/BEAR 的补充策略（后续迭代）

CHOP 分析完成后，BULL/BEAR 可做简单定性判断：
1. BULL 维度（MA防守、MA排列强度）：用股票的 MA 偏离度与收益做 Pearson 相关
2. BEAR 维度（底部放量）：用量比 × 跌幅的交互项与反弹幅度做相关
3. 不做 IC（样本 < 200），只做方向性判断：维度符号是否正确
