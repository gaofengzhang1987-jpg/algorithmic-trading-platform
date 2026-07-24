### Task 3 Report: CZSC 信号分片并行

**状态:** 完成
**日期:** 2026-07-24

---

## 产出

### 1. `tmp_out/signal_chunk.py` (新建)
- 独立脚本，接受 `--chunk N/TOTAL` 参数
- 按分片索引处理对应股票子集 (`codes[chunk_idx::total_chunks]`)
- 跳过 `< 120` bars 的新股
- 每 100 只写进度文件 (`tmp_out/signal_progress_chunk_N.txt`)
- 完成后写结果 JSON (`tmp_out/chunk_N_result.json`)
- 与 `l1_update.py` 共享相同的 `signal_config.get_config(freq="日线")` 配置
- `sys.path.insert(0, str(BASE))` 确保 `tmp_out/` 下可 import 根目录模块

### 2. `l1_update.py` (修改)
- `regenerate_signals()`: B 方案 (4 片 nohup 并行) → 失败回退 A 方案 (单线程串行)
- `_run_chunks_b()`: 启动 4 个 `nohup python3 tmp_out/signal_chunk.py --chunk N/4 &` 进程，轮询等待 result 文件，超时 600s
- `_aggregate_chunk_results()`: 汇总 4 个 result JSON 的成功数
- `_run_serial_signals()`: 原单线程逻辑，使用 `_is_new_stock()` 跳过新股，每 500 只写进度

---

## 验证

| 检查项 | 结果 |
|--------|------|
| `l1_update.py --dry-run` | 通过 (17:34:53, 0s 完成) |
| `signal_chunk.py --chunk 0/4` | CZSC 信号生成正常，~400 bars/s，000001/000008/000014 均成功处理 |
| imports (czsc, signal_config) | 全部可导入 |

---

## 关键设计决策

1. **sys.path 插入**: `signal_chunk.py` 位于 `tmp_out/` 下，直接 `import signal_config` 会失败。在 import 前加入 `sys.path.insert(0, str(BASE))` 解决
2. **回退阈值**: 单个分片失败率 > 10% 或全部超时 → 自动回退单线程
3. **进度粒度**: B 方案每 100 只，A 方案每 500 只，与任务规模匹配
4. **新股跳过**: 使用 `_is_new_stock()` (MIN_BARS=120) 而非原始 `len(df) < 30`
