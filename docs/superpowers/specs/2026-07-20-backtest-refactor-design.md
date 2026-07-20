 # 回测引擎架构重构设计

 > 日期：2026-07-20  
 > 状态：待审核  
 > 依赖：无外部依赖

 ## 目标

 将 `backtest.py`（939 行 God File）拆分为职责清晰的模块化组件，消除跨文件的重复代码和依赖倒置问题，建立统一的共享工具层。

 ## 动机：当前架构问题

 | 问题 | 表现 |
 |------|------|
 | God File | `backtest.py` 混合了信号解析、数据加载、结构缓存、ExitEngine、trade simulation、metrics、EntryFilter 副本等 10+ 个关注点 |
 | 跨文件重复 | `_parse_signal` 变体重复 5 次；`compute_metrics` 重复 3 次；`_get_stroke/_get_volume/_get_macd` 在 `backtest_phase2.py` 和 `phase4_gridsearch.py` 各写一遍 |
 | 依赖倒置 | `tmp_out/` 下 9 个脚本从 `backtest.py` import 通用工具函数（如 `load_daily`），但 `backtest.py` 本应是回测入口 |
 | 内部冗余 | `EntryFilter` + `REGIME_WEIGHTS` + `REGIME_THRESHOLDS` 同时存在于 `backtest.py` 和 `entry_filter.py` |
 | 命名分裂 | 同一 parser 叫 `_parse_signal` / `_parse` / `_p`；同一 metrics 叫 `compute_metrics` / `mets` |
 | 行为不一致 | Phase 脚本各自内联简化版 trade simulation，与主引擎 `simulate_trades()` 的 ExitEngine 状态机逻辑不同 |

 ## 架构：目标模块树

 ```
 Algorithmic Trading Platform/
 ├── core/                       # 共享工具层（所有脚本导入的唯一来源）
 │   ├── __init__.py
 │   ├── constants.py            # 信号列名 / 交易参数常量
 │   ├── data.py                 # load_daily / load_signals / get_next_trading_day / get_price_at_date
 │   ├── signal_parser.py        # parse_signal（唯一版本，统一命名）
 │   ├── signal_detector.py      # detect_all_changes
 │   ├── structure_cache.py      # load_structure_cache / lookup_structure
 │   └── metrics.py              # compute_metrics（唯一版本）
 │
 ├── backtest/                   # 回测引擎（只依赖 core + entry_filter）
 │   ├── __init__.py
 │   ├── exit_engine.py          # ExitEngine class
 │   └── runner.py               # simulate_trades / run_all / main
 │
 ├── entry_filter.py             # L2 打分（不变）
 ├── signal_engine.py            # 信号生成（不变）
 ├── strategy.py                 # 策略配置（不变）
 ├── signal_config.py            # 信号配置（不变）
 │
 ├── backtest_phase2.py          # Phase 2 单规则贡献回测（只改 import）
 ├── phase2_postfilter.py        # Phase 2 后过滤（只改 import）
 ├── phase3_combinations.py      # Phase 3 组合测试（只改 import）
 ├── phase4_gridsearch.py        # Phase 4 网格搜索（只改 import）
 │
 └── tmp_out/                    # 临时脚本（只改 import）
     ├── l1_score_all.py
     ├── l1_score_v2.py
     ├── l1_score_final.py
     ├── l1_score_buy23.py
     ├── l1_screening.py
     ├── buy1_rescore.py
     ├── buy3_phase2.py
     ├── buy23_vm.py
     ├── backtest_full.py
     ├── backtest_verify.py
     ├── optuna_chop_prep.py
     └── rebuild_backtest_input.py
 ```

 ## 调用关系图

 ```
 tmp_out/* ─────────→ core.data, core.signal_detector
 tmp_out/l1_score_all → entry_filter
 tmp_out/buy1_rescore  → entry_filter

 backtest_phase2.py  ──→ core.* + backtest.runner
 phase2_postfilter.py ──→ core.signal_parser, core.metrics
 phase3_combinations.py → core.signal_parser, core.metrics
 phase4_gridsearch.py ──→ core.* + backtest.runner

 backtest/runner.py ──→ core.* + backtest.exit_engine + entry_filter
 backtest/exit_engine.py → core.constants
 ```

 ## 各模块 API 设计

 ### core.constants

 集中管理所有被 2+ 文件引用的常量：

 ```python
 # 信号列名
 BUY1_COL  = "日线_D1B_BUY1"
 BS2_COL   = "日线_D1#SMA#21_BS2辅助V230320"
 BS3_COL   = "日线_D1#SMA#34_BS3辅助V230318"
 SANMAI_COL  = "日线_D1_三买辅助V230228"
 SANMAI2_COL = "日线_D1#SMA#34_BS3辅助V230319"
 SELL1_COL   = "日线_D1B_SELL1"
 BUY_COLS  = [BUY1_COL, BS2_COL, BS3_COL, SANMAI_COL, SANMAI2_COL]
 SELL_COLS = [SELL1_COL, BS2_COL, BS3_COL]

# 交易参数
COMMISSION = 0.001
CAPITAL_PER_TRADE = 100_000
MAX_HOLD_DAYS = 99999
STOP_LOSS_PCT = -0.08

# 入场折扣
DISCOUNT_BUY1 = 0.96
DISCOUNT_BUY2 = 0.92
DISCOUNT_BUY3 = 0.95

# 半仓超时
HALF_CUT_TIMEOUT = 30

# 数据路径
 DATA_DIR = BASE_DIR / "data"
 DAILY_DIR = DATA_DIR / "daily"
 SIGNALS_DIR = DATA_DIR / "signals"
 ```

 ### core.data

 ```python
 def load_daily(code: str) -> pd.DataFrame | None:
     """从 data/daily/{code}.parquet 加载日线数据，按 date 排序。"""

 def load_signals(code: str) -> pd.DataFrame | None:
     """从 data/signals/{code}.parquet 加载信号数据。"""

 def get_next_trading_day(date_str: str, daily: pd.DataFrame) -> pd.Timestamp | None:
     """给定 date_str 后 daily 中第一个 > date_str 的交易日。"""

 def get_price_at_date(date_ts: pd.Timestamp, daily: pd.DataFrame) -> float | None:
     """获取指定日期的开盘价。"""
 ```

 ### core.signal_parser

 ```python
 def parse_signal(val) -> dict:
     """统一信号解析，替代所有 _parse_signal / _parse / _p 变体。

     Args:
         val: 信号列的原始值（str / NaN / "0"）
     Returns:
         {"v1": str, "v2": str, "v3": str, "score": str}
         NaN 或 "0" 返回 {"v1": "", "v2": "", "v3": "", "score": "0"}
     """
 ```

 ### core.signal_detector

 ```python
 def detect_all_changes(sig_df: pd.DataFrame) -> list[dict]:
     """检测全部历史中的买点/卖点信号变化。

     Args:
         sig_df: 信号 DataFrame（含 dt, BUY_COLS, SELL_COLS）
     Returns:
         [{idx, date, type: "buy"|"sell", signal_label}, ...] 按时间排序
     """
 ```

 ### core.structure_cache

 ```python
 def load_structure_cache() -> pd.DataFrame | None:
     """加载全局结构缓存 data/reference/structure_cache.parquet。"""

 def lookup_structure(code: str, target_date_str: str) -> tuple[float, float]:
     """查缓存中 ≤ target_date 的最近结构值。

     Returns:
         (bi_low, gg_high) 或 (0, 0)
     """
 ```

 ### core.metrics

 ```python
 def compute_metrics(trades: list[dict]) -> dict:
     """计算回测绩效指标。

     Args:
         trades: [{code, signal_type, entry_date, exit_date,
                   entry_price, exit_price, return_pct, hold_days, exit_reason}, ...]
     Returns:
         {total_trades, win_rate, avg_return, total_return, max_return,
          min_return, avg_hold_days, sharpe}
     """
 ```

 ### backtest.exit_engine

 ```python
 class ExitEngine:
     """缠论结构出场引擎 + 仓位状态机。

     出场优先级（每 bar，最先触发胜出）：
       1. 结构止损：close ≤ defense → 全仓退出
       2. V 型暴跌：close < 最后上涨中枢 GG → 全仓退出
       3. FULL → HALF（顶分型 + 力度背驰 + 创新高）
       4. HALF → 回补 / 二卖确认 / 30 天超时
       5. 卖点信号：次日开盘价退出
     """

     def __init__(self, code: str, entry_price: float, entry_date: pd.Timestamp,
                  buy_type: str, struct_df: pd.DataFrame | None):
         """struct_df 由调用方传入，ExitEngine 不自行加载数据。"""

     def update_defense(self, bar_date):
         """逐 bar 更新结构止损线（底分型低点 / 中枢回调 GG）。"""

     def check_half_cut(self, bar_date, bar_high, fx_on_bar) -> dict | None:
         """FULL → HALF 条件：顶分型 + 力度背驰 + 高于前一同向笔高点。"""

     def check_buyback(self, bar_date, bar_high, half_cut_fx_high) -> bool:
         """HALF → FULL 回补：突破半仓卖出时的分型高点。"""

     def check_second_sell(self, bar_date, bar_high, half_cut_fx_high, fx_on_bar) -> bool:
         """二卖确认：半仓后出现新顶分型。"""

     def check_v_drop(self, bar_date, bar_close) -> bool:
         """V 型暴跌：close < 最后上涨中枢 GG。"""

     def rebind_defense(self, bar_date):
         """回补后重绑止损线。"""
 ```

 ### backtest.runner

 ```python
 def simulate_trades(
     code: str,
     daily: pd.DataFrame,
     changes: list[dict],
     struct_df: pd.DataFrame | None = None,
     max_trades: int = 15,
 ) -> list[dict]:
     """单只股票回测。调用方负责传数据，不内部加载文件。

     Args:
         code: 股票代码
         daily: 日线 DataFrame（已按 date 排序、reset_index）
         changes: signal_detector.detect_all_changes 的输出
         struct_df: 结构缓存 DataFrame（可选；None 时 ExitEngine 降级）
         max_trades: 单股最大交易笔数

     Returns:
         [{code, signal_type, signal_date, entry_date, exit_date,
           entry_price, exit_price, return_pct, hold_days, exit_reason}, ...]

     内部使用 ExitEngine 执行逐 bar 仓位状态机。
     """

 def run_all(
     codes: list[str] | None = None,
 ) -> pd.DataFrame:
     """批量回测入口。

     内部循环：load_daily → load_signals → detect_all_changes
     → simulate_trades → aggregate → compute_metrics
     """

 def main():
     """CLI 入口：python -m backtest.runner [codes...]"""
 ```

 ## 数据流

 ```
 run_all(codes)
   │
   └─ for each code:
        daily = core.data.load_daily(code)            # data/daily/{code}.parquet
        sig_df = core.data.load_signals(code)          # data/signals/{code}.parquet
        changes = core.signal_detector.detect_all_changes(sig_df)
        struct_df = core.structure_cache.load_structure_cache()  # 全局单次加载
        │
        └─ simulate_trades(code, daily, changes, struct_df)
             │
             └─ for each buy_event:
                  engine = ExitEngine(code, entry_price, entry_date, buy_type, struct_df)
                  │
                  │  # ─── 逐 bar 主循环（仓位状态机在 simulate_trades 内部） ───
                  for bar in window:
                      engine.update_defense(bar_date)
                      if bar_close <= engine.defense → 结构止损退出
                      if engine.check_v_drop(bar_date, bar_close) → V 型暴跌退出
                      if state == FULL and engine.check_half_cut(...) → 半仓
                      if state == HALF and engine.check_buyback(...) → 回补
                      if state == HALF and half_cut_day_count >= HALF_CUT_TIMEOUT → 半仓超时
                      if sell_exit_target reached → 卖点退出
                  │
                  │  # ─── 到期平仓 ───
                  if no exit triggered → 到期平仓
                  │
                  └─→ trade record

     └─ aggregate → pd.DataFrame → core.metrics.compute(trades) → 打印报告
 ```

 ## 迁移计划

 ### Phase 脚本改动范围

 每个 Phase 脚本只改 import 路径，不动逻辑——Python 中不存在符号级别的紧耦合问题：

 | 文件 | 改动 | 说明 |
 |------|------|------|
 | `backtest_phase2.py` | import + `simulate_filtered` 改为调 `backtest.runner.simulate_trades` | 消除内联简化版 trade loop |
 | `phase2_postfilter.py` | import `core.signal_parser`, `core.metrics` | 替代本地 `_parse` / `compute_metrics` |
 | `phase3_combinations.py` | import `core.signal_parser`, `core.metrics` | 同上 |
 | `phase4_gridsearch.py` | import `core.signal_parser`, `core.metrics`, `backtest.runner` | 替代本地 `_p` / `mets` / 内联 loop |
 | `tmp_out/l1_score_all.py` | import `from core.data import load_daily` | 替代 `from backtest import load_daily` |
 | `tmp_out/l1_score_v2.py` | import `from core.data import load_daily` | 同上 |
 | `tmp_out/l1_score_final.py` | import `from core.data import load_daily` + 从 `entry_filter` 导入 `EntryFilter`/权重 | 替代 `from backtest import EntryFilter, ...` |
 | `tmp_out/l1_score_buy23.py` | import `from core.data import load_daily` | 同上 |
 | `tmp_out/l1_screening.py` | import `from core.data import load_daily` + 从 `entry_filter` 导入 `EntryFilter` | 同上 |
 | `tmp_out/buy1_rescore.py` | import `from core.data import load_daily` + 从 `entry_filter` 导入权重 | 替代 `from backtest import EntryFilter, ...` |
 | `tmp_out/buy3_phase2.py` | import `from core.data import load_daily` | 替代 `from backtest import load_daily` |
 | `tmp_out/buy23_vm.py` | import `from core.data import load_daily`, `from core.signal_detector import detect_all_changes` | 同上 |
 | `tmp_out/backtest_full.py` | import `from core.data import load_daily` | 替代本地数据加载 |
 | `tmp_out/backtest_verify.py` | 从 `entry_filter` 导入 `REGIME_WEIGHTS` / `REGIME_THRESHOLDS` | 替代 `from backtest import` |
 | `tmp_out/optuna_chop_prep.py` | import `from core.data import load_daily` + `from entry_filter import EntryFilter` | 替代 `from backtest import load_daily` |
 | `tmp_out/rebuild_backtest_input.py` | import `from core.data import load_daily` + `from entry_filter import EntryFilter` | 替代 `from backtest import load_daily` |

 ### 删除清单

 `backtest.py` 拆分后删除以下内容：

 | 删除内容 | 原位置 | 去向 |
|---------|--------|------|
| `_parse_signal()` | L55-63 | `core/signal_parser.py` |
| `_detect_all_changes()` | L65-111 | `core/signal_detector.py` |
| `load_daily()` | L112-118 | `core/data.py` |
| `get_next_trading_day()` | L119-127 | `core/data.py` |
| `get_price_at_date()` | L128-134 | `core/data.py` |
| `_load_structure_cache()` (模块级) | L135-143 | `core/structure_cache.py` |
| `_load_structure_cache()` (函数版) | L144 | `core/structure_cache.py` |
| `_lookup_structure()` | L147-163 | `core/structure_cache.py` |
| `_get_czsc_at_date()` | L165-202 | 删除（仅 `_get_structure_stop` 使用） |
| `_get_structure_stop()` | L203-218 | 删除（已被 ExitEngine 替代） |
| `_get_structure_stop_old()` | L219-266 | 删除（已被 ExitEngine 替代） |
| `ExitEngine` class | L299-544 | `backtest/exit_engine.py` |
| `simulate_trades()` | L546-729 | `backtest/runner.py` |
| `compute_metrics()` | L731-764 | `core/metrics.py` |
| `run_all()` | L766-789 | `backtest/runner.py` |
| `main()` | L791-810 | `backtest/runner.py` |
| `REGIME_WEIGHTS` 副本 | L803-811 | 引用 `entry_filter` 即可 |
| `REGIME_THRESHOLDS` 副本 | L813-831 | 同上 |
| `_get_weights()` | L835-838 | 引用 `entry_filter._get_weights` |
| `EntryFilter` 副本 | L847 | 引用 `entry_filter` 即可 |

 ## 不变项（Non-Goals）

 - `entry_filter.py`：不修改内部逻辑，不移出任何函数
 - `signal_engine.py` / `strategy.py` / `signal_config.py` / `app.py`：不修改
 - `tmp_out/` 下脚本的业务逻辑：不修改，只修 import 路径
 - ExitEngine 出场逻辑：逐行原样迁移，不优化
 - `simulate_trades()` 仓位状态机：逐行原样迁移，不优化
 - 常量数值：不修改

 ## 验证策略

 1. **import 完整性**：`python -c "from core.data import load_daily; from backtest.runner import simulate_trades"` 零 import error
 2. **数据加载**：`load_daily("000001")` 返回与原 `backtest.load_daily()` 相同的 DataFrame
 3. **信号解析**：`parse_signal(old_val)` 返回与原 `_parse_signal(old_val)` 相同的 dict
 4. **Phase 脚本**：每个 Phase 脚本 `python phaseX.py --help` 或 dry-run 不报 ImportError
 5. **端到端**：`python -m backtest.runner 000001` 输出与原 `python backtest.py 000001` 相同的 trades + metrics
 6. **批量回归**：`python -m backtest.runner`（前 100 只股票）输出指标与原版偏差 < 1%（允许浮点舍入）

 ## 实施顺序

 1. 创建 `core/` 包，迁移常量 + 数据加载 + 信号解析器（0 依赖）
 2. 迁移 signal_detector（依赖 core.constants + core.signal_parser）
 3. 迁移 metrics（0 依赖）
 4. 迁移 structure_cache（依赖 core.constants）
 5. 创建 `backtest/` 包，迁移 ExitEngine（依赖 core.constants）
 6. 迁移 runner（依赖 core.* + backtest.exit_engine + entry_filter）
 7. 修所有 Phase 脚本的 import
 8. 删除 `backtest.py`
 9. 端到端回归测试
