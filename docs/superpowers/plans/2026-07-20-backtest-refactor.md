 # 回测引擎架构重构实施计划

 > **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

 **Goal:** 将 `backtest.py`（939 行 God File）拆分为 `core/`（共享工具层）+ `backtest/`（回测引擎）两个包，消除跨文件重复和依赖倒置，不改动任何业务逻辑。

 **Architecture:** 提取 `core/` 下的 constants、data、signal_parser、signal_detector、metrics、structure_cache 六个模块作为共享基础层；提取 ExitEngine 和 runner 到 `backtest/` 包；Phase 脚本只改 import 路径。

 **Tech Stack:** Python 3.12, pandas, numpy, czsc, pathlib

 ## Global Constraints

- 不修改任何业务逻辑——函数体和类方法逐行原样迁移
- 不修改 `entry_filter.py`、`signal_engine.py`、`strategy.py`、`signal_config.py`、`app.py`
- 不修改 `tmp_out/` 下脚本的内部逻辑，只改 import 路径
- 常量数值不修改
- 每一个 Task 完成后必须通过其验证步骤，否则不进入下一个 Task
- 提交粒度：每个 Task 一个 commit

 ---

 ### Task 1: core/constants.py + core/signal_parser.py + core/__init__.py

 **Files:**
 - Create: `core/__init__.py`
 - Create: `core/constants.py`
 - Create: `core/signal_parser.py`

 **Interfaces:**
 - Consumes: nothing（零依赖）
 - Produces: `core.constants` (COMMISSION, CAPITAL_PER_TRADE, etc.), `core.signal_parser.parse_signal`

 - [ ] **Step 1: 创建 core/ 目录和空的 __init__.py**

 ```bash
 mkdir -p core backtest
 touch core/__init__.py backtest/__init__.py
 ```

 - [ ] **Step 2: 创建 core/constants.py — 从 backtest.py L27-53 无损移出**

 ```python
 """回测系统共享常量 — 信号列名 / 交易参数 / 路径。"""
 from pathlib import Path

 BASE_DIR = Path(__file__).resolve().parent.parent
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
 BUY_COLS = [BUY1_COL, BS2_COL, BS3_COL, SANMAI_COL, SANMAI2_COL]
 SELL_COLS = [SELL1_COL, BS2_COL, BS3_COL]
 ```

 - [ ] **Step 3: 创建 core/signal_parser.py — 从 backtest.py L55-63 无损移出，改名 `parse_signal`**

 ```python
 """统一信号解析器 — 项目中唯一版本的 parse_signal。"""
 import pandas as pd


 def parse_signal(val):
     """解析 CZSC 信号列的原始值。

     Args:
         val: 信号列原始值（str / NaN / "0"）
     Returns:
         {"v1": str, "v2": str, "v3": str, "score": str}
         NaN 或 "0" 返回 {"v1": "", "v2": "", "v3": "", "score": "0"}
     """
     if pd.isna(val) or str(val) == "0":
         return {"v1": "", "v2": "", "v3": "", "score": "0"}
     parts = str(val).rsplit("_", 3)
     return {"v1": parts[0] if len(parts) >= 4 else "",
             "v2": parts[1] if len(parts) >= 4 else "",
             "v3": parts[2] if len(parts) >= 4 else "",
             "score": parts[3] if len(parts) >= 4 else "0"}
 ```

 - [ ] **Step 4: 验证 import 完整性**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 python3 -c "from core.constants import COMMISSION, BUY_COLS; from core.signal_parser import parse_signal; print('Task 1 OK')"
 ```
 Expected: 打印 `Task 1 OK`，无 ImportError

 - [ ] **Step 5: Commit**

 ```bash
 git add core/__init__.py core/constants.py core/signal_parser.py
 git commit -m "refactor: extract core/constants and core/signal_parser from backtest.py"
 ```

 ---

 ### Task 2: core/data.py

 **Files:**
 - Create: `core/data.py`

 **Interfaces:**
 - Consumes: `core.constants`(DAILY_DIR, SIGNALS_DIR)
 - Produces: `core.data.load_daily`, `core.data.load_signals`, `core.data.get_next_trading_day`, `core.data.get_price_at_date`

 - [ ] **Step 1: 创建 core/data.py — 从 backtest.py L112-134 无损移出，新增 load_signals**

 ```python
 """日线和信号数据加载。"""
 from typing import Optional

 import pandas as pd
 from core.constants import DAILY_DIR, SIGNALS_DIR


 def load_daily(code: str) -> pd.DataFrame | None:
     """从 data/daily/{code}.parquet 加载日线数据。"""
     p = DAILY_DIR / f"{code}.parquet"
     if not p.exists():
         return None
     return pd.read_parquet(p)


 def load_signals(code: str) -> pd.DataFrame | None:
     """从 data/signals/{code}.parquet 加载信号数据。"""
     p = SIGNALS_DIR / f"{code}.parquet"
     if not p.exists():
         return None
     return pd.read_parquet(p)


 def get_next_trading_day(date_str: str, daily: pd.DataFrame) -> Optional[pd.Timestamp]:
     """给定 date_str 后 daily 中第一个 > date_str 的交易日。"""
     sig_date = pd.Timestamp(date_str)
     mask = daily["date"] > sig_date
     future = daily.loc[mask, "date"]
     if future.empty:
         return None
     return future.iloc[0]


 def get_price_at_date(date_ts: pd.Timestamp, daily: pd.DataFrame) -> Optional[float]:
     """获取指定日期的开盘价。"""
     row = daily[daily["date"] == date_ts]
     if row.empty:
         return None
     return float(row["open"].iloc[0])
 ```

 - [ ] **Step 2: 验证 import 完整性**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 python3 -c "from core.data import load_daily, load_signals; print('Task 2 OK')"
 ```
 Expected: `Task 2 OK`

 - [ ] **Step 3: 验证数据加载 — 与原版对比**

 ```bash
 python3 -c "
 from core.data import load_daily
 import sys; sys.path.insert(0,'.')
 from backtest import load_daily as old
 df_new = load_daily('000001')
 df_old = old('000001')
 assert df_new is not None, 'load_daily returned None'
 assert df_old is not None, 'old load_daily returned None'
 assert len(df_new) == len(df_old), f'diff lengths: {len(df_new)} vs {len(df_old)}'
 print(f'load_daily OK: {len(df_new)} rows')
 "
 ```
 Expected: `load_daily OK: NNNN rows`

 - [ ] **Step 4: Commit**

 ```bash
 git add core/data.py
 git commit -m "refactor: extract core/data.py from backtest.py"
 ```

 ---

 ### Task 3: core/signal_detector.py + core/metrics.py

 **Files:**
 - Create: `core/signal_detector.py`
 - Create: `core/metrics.py`

 **Interfaces:**
 - Consumes: `core.constants`(BUY_COLS, SELL_COLS), `core.signal_parser.parse_signal`
 - Produces: `core.signal_detector.detect_all_changes`, `core.metrics.compute_metrics`

 - [ ] **Step 1: 创建 core/signal_detector.py — 从 backtest.py L65-111 无损移出，改名 `detect_all_changes`**

 ```python
 """全局信号变化检测 — 检测历史中所有买点/卖点信号变化。"""
 from core.constants import BUY1_COL, BS2_COL, BS3_COL, SANMAI_COL, SANMAI2_COL, BUY_COLS, SELL_COLS
 from core.signal_parser import parse_signal


 def detect_all_changes(sig_df) -> list[dict]:
     """检测全部历史中的买点/卖点信号变化。

     Args:
         sig_df: 信号 DataFrame（含 dt, BUY_COLS, SELL_COLS）
     Returns:
         [{idx, date, type: "buy"|"sell", signal_label}, ...] 按时间排序
     """
     buy_cols = [c for c in [BUY1_COL, BS2_COL, BS3_COL, SANMAI_COL, SANMAI2_COL] if c in sig_df.columns]
     sell_cols = [c for c in SELL_COLS if c in sig_df.columns]
     changes = []

     # 检测买点变化
     for col in buy_cols:
         for i in range(1, len(sig_df)):
             old_r = str(sig_df.iloc[i - 1][col])
             new_r = str(sig_df.iloc[i][col])
             if old_r == new_r:
                 continue
             new_p = parse_signal(new_r)
             if any(k in new_p["v1"] for k in ["一买", "二买", "三买"]):
                 dt = sig_df.iloc[i]["dt"]
                 sig_label = f"{new_p['v1']}({col.split('_')[2][:8] if '_' in col else col[:8]})"
                 changes.append({
                     "idx": i, "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                     "type": "buy", "signal_label": sig_label,
                 })

     # 检测卖点变化
     for col in sell_cols:
         for i in range(1, len(sig_df)):
             old_r = str(sig_df.iloc[i - 1][col])
             new_r = str(sig_df.iloc[i][col])
             if old_r == new_r:
                 continue
             new_p = parse_signal(new_r)
             if any(k in new_p["v1"] for k in ["一卖", "二卖", "三卖"]):
                 dt = sig_df.iloc[i]["dt"]
                 sig_label = new_p["v1"]
                 changes.append({
                     "idx": i, "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                     "type": "sell", "signal_label": sig_label,
                 })

     changes.sort(key=lambda x: x["date"])
     return changes
 ```

 - [ ] **Step 2: 创建 core/metrics.py — 从 backtest.py L731-764 无损移出**

 ```python
 """回测绩效指标计算 — 唯一版本的 compute_metrics。"""
 import numpy as np


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
 ```

 - [ ] **Step 3: 验证 import 完整性**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 python3 -c "from core.signal_detector import detect_all_changes; from core.metrics import compute_metrics; print('Task 3 OK')"
 ```
 Expected: `Task 3 OK`

 - [ ] **Step 4: Commit**

 ```bash
 git add core/signal_detector.py core/metrics.py
 git commit -m "refactor: extract core/signal_detector and core/metrics from backtest.py"
 ```

 ---

 ### Task 4: core/structure_cache.py

 **Files:**
 - Create: `core/structure_cache.py`

 **Interfaces:**
 - Consumes: `core.constants`(BASE_DIR)
 - Produces: `core.structure_cache.load_structure_cache`, `core.structure_cache.lookup_structure`

 - [ ] **Step 1: 创建 core/structure_cache.py — 从 backtest.py L135-163 无损移出**

 ```python
 """结构缓存加载 — BI 低点 + GG 高点预计算缓存。"""
 import pandas as pd
 from core.constants import BASE_DIR


 _structure_cache = None
 _struct_path = BASE_DIR / "data" / "reference" / "structure_cache.parquet"
 _structure_cache = pd.read_parquet(_struct_path) if _struct_path.exists() else None


 def load_structure_cache():
     return _structure_cache


 def lookup_structure(code, target_date_str):
     """查缓存中 <= target_date 的最近结构值。

     Returns:
         (bi_low, gg_high) 或 (0, 0)
     """
     sc = load_structure_cache()
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
 ```

 - [ ] **Step 2: 验证 import 完整性**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 python3 -c "from core.structure_cache import load_structure_cache, lookup_structure; print('Task 4 OK')"
 ```
 Expected: `Task 4 OK`

 - [ ] **Step 3: Commit**

 ```bash
 git add core/structure_cache.py
 git commit -m "refactor: extract core/structure_cache from backtest.py"
 ```

 ---

 ### Task 5: backtest/exit_engine.py

 **Files:**
 - Create: `backtest/exit_engine.py`

 **Interfaces:**
 - Consumes: `core.constants`(DISCOUNT_BUY1, DISCOUNT_BUY2, DISCOUNT_BUY3, STOP_LOSS_PCT, HALF_CUT_TIMEOUT)
 - Produces: `backtest.exit_engine.ExitEngine`

 - [ ] **Step 1: 创建 backtest/exit_engine.py — 从 backtest.py L299-544 逐行原样迁移**

 提取指令：截取 backtest.py 第 299-544 行中 `class ExitEngine:` 的完整定义（含所有方法），粘贴到新文件。头部添加 import：

 ```python
 """缠论结构出场引擎 — 运作仓位状态机的信号判断器。"""
 import numpy as np
 import pandas as pd
 from core.constants import (
     DISCOUNT_BUY1, DISCOUNT_BUY2, DISCOUNT_BUY3,
     STOP_LOSS_PCT, HALF_CUT_TIMEOUT,
 )


 class ExitEngine:
     # ... 从 backtest.py L299-544 原样复制 ...
 ```

 > 注意：ExitEngine 内部的 `self.struct_df` 由构造函数接收，不再自行加载。删除 `_load_structure_cache(code)` 的回退调用（如果有的话）。

 - [ ] **Step 2: 验证 ExitEngine 可导入且结构完整**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 python3 -c "
 from backtest.exit_engine import ExitEngine
 methods = [m for m in dir(ExitEngine) if not m.startswith('_')]
 expected = ['check_buyback','check_half_cut','check_second_sell','check_v_drop','rebind_defense','update_defense']
 for m in expected:
     assert m in methods, f'Missing method: {m}'
 print('ExitEngine methods:', methods)
 print('Task 5 OK')
 "
 ```
 Expected: 列出 6 个公有方法，打印 `Task 5 OK`

 - [ ] **Step 3: Commit**

 ```bash
 git add backtest/exit_engine.py
 git commit -m "refactor: extract backtest/exit_engine.py from backtest.py"
 ```

 ---

 ### Task 6: backtest/runner.py

 **Files:**
 - Create: `backtest/runner.py`

 **Interfaces:**
 - Consumes: `core.*`(all), `backtest.exit_engine.ExitEngine`, `entry_filter.REGIME_WEIGHTS`, `entry_filter.REGIME_THRESHOLDS`
 - Produces: `backtest.runner.simulate_trades`, `backtest.runner.run_all`, `backtest.runner.main`

 - [ ] **Step 1: 创建 backtest/runner.py — 从 backtest.py 剩余部分组合而成**

 头部的 imports + logger 设置：

 ```python
 #!/usr/bin/env python3
 """缠论信号回测引擎 — 批量回测入口。"""
 import logging
 from pathlib import Path
 from typing import Optional

 import numpy as np
 import pandas as pd

 from core.constants import (
     COMMISSION, CAPITAL_PER_TRADE, MAX_HOLD_DAYS, STOP_LOSS_PCT,
     HALF_CUT_TIMEOUT, SIGNALS_DIR,
 )
 from core.data import load_daily, load_signals, get_next_trading_day, get_price_at_date
 from core.signal_detector import detect_all_changes
 from core.structure_cache import load_structure_cache, lookup_structure
 from core.metrics import compute_metrics
 from backtest.exit_engine import ExitEngine
 from entry_filter import REGIME_WEIGHTS, REGIME_THRESHOLDS

 logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
 logger = logging.getLogger("backtest")
 ```

 然后从 backtest.py 移入：
 - `simulate_trades()`（L546-729）—— 改动：删除函数内部的 `load_daily(code)` 和 `sig_df = pd.read_parquet(...)`，改为接收参数
 - `run_all()`（L766-789）—— 增加 `load_signals` 调用
 - `main()`（L791-810）

 `simulate_trades` 签名改为：

 ```python
 def simulate_trades(
     code: str,
     daily: pd.DataFrame,
     changes: list[dict],
     struct_df: pd.DataFrame | None = None,
     max_trades: int = 15,
 ) -> list[dict]:
 ```

 函数体内删除以下行：
 - `daily = load_daily(code)`（约 L547-549）
 - `sig_path = SIGNALS_DIR / f"{code}.parquet"` 及其后的 `pd.read_parquet` 和 `_detect_all_changes`（约 L551-557）

 替换为直接使用参数 `daily` 和 `changes`。

 `run_all` 改为内部循环做文件 IO：

 ```python
 def run_all(codes: list[str] | None = None) -> pd.DataFrame:
     if codes is None:
         codes = sorted(p.stem for p in SIGNALS_DIR.glob("*.parquet"))

     struct_cache = load_structure_cache()
     all_trades = []
     for i, code in enumerate(codes):
         try:
             daily = load_daily(code)
             if daily is None:
                 continue
             sig_df = load_signals(code)
             if sig_df is None:
                 continue
             changes = detect_all_changes(sig_df)
             trades = simulate_trades(code, daily, changes, struct_cache)
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
 ```

 - [ ] **Step 2: 验证 runner 可导入且结构完整**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 python3 -c "
 from backtest.runner import simulate_trades, run_all, main
 print('Task 6 OK')
 "
 ```
 Expected: `Task 6 OK`

 - [ ] **Step 3: Commit**

 ```bash
 git add backtest/runner.py
 git commit -m "refactor: extract backtest/runner.py from backtest.py"
 ```

 ---

 ### Task 7: Phase 脚本迁移（16 文件）

 **Files:**
 - Modify: `backtest_phase2.py`, `phase2_postfilter.py`, `phase3_combinations.py`, `phase4_gridsearch.py`
 - Modify: `tmp_out/l1_score_all.py`, `tmp_out/l1_score_v2.py`, `tmp_out/l1_score_final.py`, `tmp_out/l1_score_buy23.py`, `tmp_out/l1_screening.py`, `tmp_out/buy1_rescore.py`, `tmp_out/buy3_phase2.py`, `tmp_out/buy23_vm.py`, `tmp_out/backtest_full.py`, `tmp_out/backtest_verify.py`, `tmp_out/optuna_chop_prep.py`, `tmp_out/rebuild_backtest_input.py`

 **Interfaces:**
 - Consumes: `core.*`, `backtest.runner`, `entry_filter`
 - Produces: 每个脚本 import 修复后零 ImportError

 - [ ] **Step 1: 修改 backtest_phase2.py 的 import**

 ```diff
 - from backtest import (
 -     load_daily, _parse_signal, _detect_all_changes,
 -     get_next_trading_day, get_price_at_date, compute_metrics,
 -     SIGNALS_DIR, COMMISSION, CAPITAL_PER_TRADE, MAX_HOLD_DAYS, STOP_LOSS_PCT,
 - )
 + from core.data import load_daily, get_next_trading_day, get_price_at_date
 + from core.signal_parser import parse_signal
 + from core.signal_detector import detect_all_changes
 + from core.metrics import compute_metrics
 + from core.constants import SIGNALS_DIR, COMMISSION, CAPITAL_PER_TRADE, MAX_HOLD_DAYS, STOP_LOSS_PCT

  # 所有 _parse_signal(...) 调用替换为 parse_signal(...)
  # 所有 _detect_all_changes(...) 调用替换为 detect_all_changes(...)
 ```

 - [ ] **Step 2: 修改 phase2_postfilter.py 的 import**

 ```diff
 - def _parse(val): ...  # 删除本地定义
 - def compute_metrics(trades_list): ...  # 删除本地定义
 + from core.signal_parser import parse_signal  # 替代 _parse
 + from core.metrics import compute_metrics
 #
 # 所有 _parse(...) 调用替换为 parse_signal(...)
 ```

 - [ ] **Step 3: 修改 phase3_combinations.py 的 import**

 ```diff
 - def _parse(val): ...      # 删除
 - def compute_metrics(...): ...  # 删除
 + from core.signal_parser import parse_signal
 + from core.metrics import compute_metrics
 #
 # 所有 _parse(...) → parse_signal(...)
 ```

 - [ ] **Step 4: 修改 phase4_gridsearch.py 的 import**

 ```diff
 - def _p(val): ...           # 删除
 - def mets(trades): ...      # 删除
 + from core.signal_parser import parse_signal as _p  # 别名兼容
 + from core.metrics import compute_metrics as mets    # 别名兼容
 ```

 - [ ] **Step 5: 修改 tmp_out/ 下所有脚本（批量）**

 对每个脚本，将以下 import 换源：

 | 旧 import | 新 import |
 |----------|----------|
 | `from backtest import load_daily` | `from core.data import load_daily` |
 | `from backtest import _parse_signal` | `from core.signal_parser import parse_signal` |
 | `from backtest import _detect_all_changes` | `from core.signal_detector import detect_all_changes` |
 | `from backtest import EntryFilter` | `from entry_filter import EntryFilter` |
 | `from backtest import REGIME_WEIGHTS` | `from entry_filter import REGIME_WEIGHTS` |
 | `from backtest import REGIME_THRESHOLDS` | `from entry_filter import REGIME_THRESHOLDS` |

 同时将代码中的 `_parse_signal(...)` 替换为 `parse_signal(...)`，`_detect_all_changes(...)` 替换为 `detect_all_changes(...)`。

 - [ ] **Step 6: 验证所有 Phase 脚本导入无错误**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 for f in backtest_phase2.py phase2_postfilter.py phase3_combinations.py phase4_gridsearch.py; do
   echo -n "$f: "
   python3 -c "import ast; ast.parse(open('$f').read()); print('syntax OK')"
 done
 ```
 Expected: 4 files × `syntax OK`

 对 tmp_out/ 下脚本同理（跳过已知有运行时依赖的）：

 ```bash
 for f in tmp_out/l1_score_all.py tmp_out/optuna_chop_prep.py tmp_out/buy1_rescore.py tmp_out/rebuild_backtest_input.py tmp_out/backtest_verify.py tmp_out/buy3_phase2.py; do
   echo -n "$f: "
   python3 -c "import ast; ast.parse(open('$f').read()); print('syntax OK')"
 done
 ```

 - [ ] **Step 7: Commit**

 ```bash
 git add -u
 git commit -m "refactor: migrate all phase scripts to core/* and backtest/* imports"
 ```

 ---

 ### Task 8: 删除 backtest.py + 端到端回归验证

 **Files:**
 - Delete: `backtest.py`

 - [ ] **Step 1: 删除 backtest.py**

 ```bash
 git rm backtest.py
 ```

 - [ ] **Step 2: 全局搜索确保没有残留 import**

 ```bash
 rg "from backtest import|import backtest" --type py -l | grep -v __pycache__ | grep -v node_modules
 ```
 Expected: 空输出（如果还有文件残留引用，修完再继续）

 - [ ] **Step 3: 端到端验证 — 单只股票回测**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 python3 -c "
 from core.data import load_daily, load_signals
 from core.signal_detector import detect_all_changes
 from core.metrics import compute_metrics
 from core.structure_cache import load_structure_cache
 from backtest.runner import simulate_trades

 code = '000001'
 daily = load_daily(code)
 sig_df = load_signals(code)
 changes = detect_all_changes(sig_df)
 struct_df = load_structure_cache()

 trades = simulate_trades(code, daily, changes, struct_df)
 m = compute_metrics(trades)
 print(f'000001: {m[\"total_trades\"]} trades, WR={m[\"win_rate\"]}%, avg_ret={m[\"avg_return\"]}%, sharpe={m[\"sharpe\"]}')
 print('End-to-end OK')
 "
 ```
 Expected: 输出 000001 的回测指标，打印 `End-to-end OK`

 - [ ] **Step 4: 批量回归验证 — 前 10 只股票**

 ```bash
 cd /Users/hz/Desktop/Algorithmic\ Trading\ Platform
 python3 -m backtest.runner 000001 000002 000004 000005 000006 000007 000008 000009 000010 000011
 ```
 Expected: 打印 `回测完成: N 只股票, M 笔交易` 和绩效指标，无异常/崩溃

 - [ ] **Step 5: Commit**

 ```bash
 git add -u
 git commit -m "refactor: delete backtest.py, verify end-to-end"
 ```

 ---

 ## 实施顺序依赖

 ```
 Task1 (constants+parser) ─┬─→ Task2 (data) ──────────┐
                            ├─→ Task3 (detector+metrics) ├─→ Task5 (exit_engine)
                            ├─→ Task4 (structure_cache) ├─→ Task6 (runner)
                            └───────────────────────────┘        │
                                                          Task7 (phase scripts)
                                                               │
                                                          Task8 (delete+verify)
 ```

 Task 1-4 可在同一会话中批量执行（无相互依赖），Task 5 依赖 Task 1，Task 6 依赖 Task 1-5。
