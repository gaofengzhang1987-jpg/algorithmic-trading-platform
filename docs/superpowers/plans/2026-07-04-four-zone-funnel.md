# 四级选股漏斗 + 监控塔 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建四级选股漏斗（L1→L4）和区域二监控塔，所有信号计算调用 czsc 原生 API。

**Architecture:** 全市场 → L1 全量沉淀区(一买/二买/三买) → L2 策略一区 → L3 策略二区 → L4 购买建议区。非 L1 池股票监控一卖/二卖/三卖，L3/L4 细颗粒度监控。

**Tech Stack:** czsc (generate_czsc_signals, CzscStrategyBase), Streamlit, Pandas, Plotly

---

## 文件结构

```
Algorithmic Trading Platform/
├── signal_config.py          # [修改] 扩展信号序列（加二买/三买/卖点）
├── zone1_deposition.py       # [新建] L1 全量沉淀区
├── zone2_pattern.py          # [新建] L2 策略一区（框架，空规则）
├── zone3_watchlist.py        # [新建] L3 策略二区（框架，空规则）  
├── zone4_recommend.py        # [新建] L4 购买建议区
├── monitor.py                # [新建] 区域二监控塔（框架）
├── app.py                    # [修改] 6 页签仪表盘
└── data/zones/               # [新建] 各区结果缓存目录
    ├── L1_deposition.parquet
    ├── L2_pattern.parquet
    ├── L3_watchlist.parquet
    ├── L4_recommend.parquet
    └── monitor_sell.parquet
```

### 文件职责

| 文件 | 单次职责 | 依赖 czsc API |
|---|---|---|
| signal_config.py | 定义所有 czsc 信号序列 | get_signals_config |
| zone1_deposition.py | L1 筛选：从信号 Parquet 中找含买点的股票 | 读 stateful 信号列 |
| zone2_pattern.py | L2 筛选：对 L1 结果应用策略一区规则 | CzscStrategyBase (框架) |
| zone3_watchlist.py | L3 筛选：对 L2 结果叠加策略二区 | CzscStrategyBase (框架) |
| zone4_recommend.py | L4 终选：输出最高置信度买点 | 排序/评分 |
| monitor.py | 扫描非 L1 股票的卖点 + L3/L4 细监 | generate_czsc_signals |
| app.py | Streamlit 6 页签（漏斗概览 + 4级区 + 监控）| KlineChart |

---

## 任务列表

### 阶段 1: 信号配置扩展

- [ ] 1.1 查 cxt_first_buy 全部笔数变体，补充 11/13/15/17/19/21 笔
- [ ] 1.2 添加二买信号：`日线_D1#SMA#21_BS2辅助V230320_二买_任意_任意_0`
- [ ] 1.3 添加三买信号：`日线_D1#SMA#34_BS3辅助V230318_三买_任意_任意_0`
- [ ] 1.4 添加三买辅助信号：`日线_D1_三买辅助V230228_三买_{6/8/10/12/14}笔_任意_0`
- [ ] 1.5 添加一卖（全部笔数）+ 二卖 + 三卖信号
- [ ] 1.6 验证 `get_config()` 返回正确的信号配置数
- [ ] 1.7 重启全量信号计算，覆盖扩展的信号集

### 阶段 2: 四级漏斗模块

- [ ] 2.1 创建 `data/zones/` 目录
- [ ] 2.2 实现 `zone1_deposition.py` — 扫描所有信号 Parquet，输出含买点状态变化的股票
- [ ] 2.3 实现 `zone2_pattern.py` — L1 → L2，调用策略一区（框架空规则：先全部通过）
- [ ] 2.4 实现 `zone3_watchlist.py` — L2 → L3，调用策略二区（框架空规则：先全部通过）
- [ ] 2.5 实现 `zone4_recommend.py` — L3 → L4，按信号置信度/频率评分排名
- [ ] 2.6 编写 `run_zones.py` 一键执行 L1→L4 全流程

### 阶段 3: 监控塔

- [ ] 3.1 实现 `monitor.py` — 扫描非 L1 股票找卖点信号
- [ ] 3.2 添加 L3/L4 细颗粒度监控框架（占位，规则后续定义）

### 阶段 4: 仪表盘

- [ ] 4.1 更新 app.py 为 6 页签：漏斗概览 / L1 / L2 / L3 / L4 / 监控
- [ ] 4.2 漏斗概览页：显示 L1→L4 每层股票数量 + 柱状图
- [ ] 4.3 L1-L4 各页签：显示对应区域的股票表格
- [ ] 4.4 监控页签：卖点股票列表 + L3/L4 细监状态
- [ ] 4.5 使用 czsc `KlineChart` 代替手写 Plotly（个股详情）

### 阶段 5: 策略框架

- [ ] 5.1 策略一区框架：占位类，包含 `apply(df) -> df` 接口（当前返回原 df）
- [ ] 5.2 策略二区框架：占位类，同上
- [ ] 5.3 添加参数化配置入口（JSON/YAML），供后续回测调优

### 阶段 6: 验证

- [ ] 6.1 等信号全量计算完成后，运行 `run_zones.py` 验证全流程
- [ ] 6.2 检查各区结果合理性（L1 约 1000-2000 只，L2≈L1，L3≈L2，L4≤100 只）
- [ ] 6.3 Streamlit 各页签内容正常渲染

---

## 关键设计决策

1. **信号格式**: 所有信号字符串 7 部分 `_` 分隔，从 czsc 函数 docstring 提取
2. **状态变化检测**: 信号触发 = 相邻两行状态不同，而非单行状态值
3. **策略框架**: 策略一区/二区初始为全通过，后续通过回测填入规则
4. **缓存策略**: 各区结果存为 Parquet 在 `data/zones/`，避免每次重复计算
