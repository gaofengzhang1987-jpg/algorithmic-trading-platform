<!-- 2026-08-04 新增，2026-08-06 重写 -->
# Qlib ML 增强预测模块

> 更新时间：2026-08-06  
> 当前版本：V2 — 信号质量分类器

## 一、版本演进

| 版本 | 方案 | 结果 | 状态 |
|------|------|------|------|
| V1 | 45 维量价因子 → 预测 20 日 absolute return | corr=-0.078（无效） | [已废弃 2026-08-06] |
| **V2** | **20 维信号特征 → 预测买点信号质量（5% 盈利概率）** | **AUC=0.589** | **当前** |

V1 失败根因：45 维纯量价因子中，跟 RPS 重叠的 15 维被 L4 composite 的其他维度稀释，不重叠的 30 维在 630 万日频样本下被噪声淹没。

V2 思路：不再预测"任意一天的未来收益"，改为预测"这个买点信号值不值得跟"。标签与 L4 的使用场景完全对齐。

## 二、文件结构

```
qlib_ml/
├── trainer.py              # V1：45 维因子回归训练（保留，备用）
├── predictor.py            # V1：45 维预测器（保留，备用）
├── features.py             # 因子提取（45/62/71 维，trainer 和 predictor 共用）
├── signal_trainer.py       # V2：信号质量分类器训练（当前）
├── signal_predictor.py     # V2：信号质量预测器（当前，已接入 L4）
├── quarterly_retrain.py    # V1 季度重训脚本（已加固 corr<0 拒绝替换）
├── models/
│   ├── lgb_model.txt       # V1 45 维模型（Backup）
│   ├── lgb_model_signal.txt      # V2 信号质量模型（当前活跃）
│   └── lgb_model_signal_config.json
│   └── cache/              # V1 因子缓存
└── models/backups/          # 历史模型备份
```

## 三、V2 信号质量分类器

### 3.1 标签定义

```
对每个 czsc 买点信号（一买/二买/三买）：
  取信号日后 20 个交易日的最高收盘价
  if max_close / signal_close - 1 >= 5%: label = 1（好信号）
  else:                                  label = 0（坏信号）
```

### 3.2 特征设计（20 维）

| 类别 | 维度 | 说明 |
|------|------|------|
| 收益动量 | 3 | 5/10/20 日涨跌幅 |
| MA 偏离 | 3 | 5/20/60 日均线偏离度 |
| 波动率 | 2 | 10/20 日 std/close |
| ATR 比率 | 1 | ATR(14) / close |
| 高低位 | 2 | 距 60 日高点%、60 日高低区间位置 |
| 量价关系 | 4 | 当日量比、短期均量变化、10 日价量相关、放量阳线标志 |
| 趋势强度 | 2 | 10/20 日上涨天数占比 |
| Regime | 3 | BULL/BEAR/CHOP one-hot |

与 L4 现有维度（L2 结构分、RPS 动量分）零重叠。

### 3.3 训练参数

| 参数 | 值 |
|------|-----|
| `objective` | binary |
| `num_leaves` | 15 |
| `learning_rate` | 0.02 |
| `min_data_in_leaf` | 30 |
| `num_boost_round` | 300 |
| `early_stopping` | 30 |
| 训练集 | 2024-07 → 2025-12（80K 信号，三态均衡） |
| 验证集 | 2026-01 → 2026-07（12K 信号） |

### 3.4 验证结果

| 指标 | 值 |
|------|-----|
| AUC | 0.589 |
| Precision@top20% | 52.2%（vs 基线 43.1%） |
| 主导特征 | ATR 比率、波动率、量比 |

AUC 0.589 在金融信号分类中属于可用水平。BEAR 下正样本率 50.3%（高于 BULL 的 40.3%），说明熊市反弹的可预测性更强。

### 3.5 重训练

信号模型训练快速（~6 分钟全量），数据来源为 `data/signals/*.parquet`。信号文件随 L1 数据更新流水线自动更新（`run_signal_parallel.py`）。

```bash
cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
python3 qlib_ml/signal_trainer.py
```

## 四、集成点

```
L1 CZSC 信号 → L2 EntryFilter → L3 质量过滤
                                      │
                    ┌─────────────────┘
                    ▼
              SignalQlibPredictor.score()  ← qlib_ml/signal_predictor.py
                    │
                    ▼
              L4Ranker.rank()             ← composite 含 w_qlib=0.15
                    │
                    ▼
              L4 排名输出 → Backtest
```

zone4_regime.py 中当前使用 `SignalQlibPredictor`，L4 composite 公式不变：

$$composite = 0.50 \cdot nL2 + 0.25 \cdot nStockRPS + 0.25 \cdot nQlib$$

## 变更记录

| 日期 | 变更 |
|------|------|
| 2026-08-06 | 权重调整为 0.50/0.25/0.00/0.25（回测验证） |
| 2026-08-06 | V2：切换为信号质量分类器（AUC=0.589），替代 V1 的 45 维回归（corr=-0.078） |
| 2026-08-05 | 接入 QlibPredictor：qlib_score 从常数 0.5 改为 LightGBM 预测 |
| 2026-08-04 | 初始版本：45 维因子回归训练 |
