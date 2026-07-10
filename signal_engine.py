#!/usr/bin/env python3
"""czsc 信号计算引擎 — 支持日线/周线/15分钟/30分钟多级别。"""

import logging, time
from pathlib import Path
import pandas as pd
from czsc import RawBar, Freq, generate_czsc_signals
from signal_config import SIGNALS_BY_FREQ, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signal_engine")

BASE_DIR = Path(__file__).parent

FREQ_CONFIG = {
    '日线': {'data_dir': BASE_DIR/"data"/"daily", 'signals_dir': BASE_DIR/"data"/"signals", 'freq': Freq.D, 'init_n': 500, 'min_bars': 30, 'vol_col': 'volume'},
    '周线': {'data_dir': BASE_DIR/"data"/"weekly", 'signals_dir': BASE_DIR/"data"/"signals_weekly", 'freq': Freq.W, 'init_n': 50, 'min_bars': 20, 'vol_col': 'volume'},
    '15分钟': {'data_dir': BASE_DIR/"data"/"min15", 'signals_dir': BASE_DIR/"data"/"signals_15min", 'freq': Freq.F15, 'init_n': 500, 'min_bars': 50, 'vol_col': 'volume'},
    '30分钟': {'data_dir': BASE_DIR/"data"/"min30", 'signals_dir': BASE_DIR/"data"/"signals_30min", 'freq': Freq.F30, 'init_n': 300, 'min_bars': 50, 'vol_col': 'volume'},
}

SDT = "20200101"

def _load_data(code, data_dir):
    p = data_dir / f"{code}.parquet"
    if not p.exists(): return None
    return pd.read_parquet(p)

def _df_to_bars(df, freq, vol_col):
    bars = []
    for i, (_, row) in enumerate(df.iterrows()):
        bars.append(RawBar(symbol=str(row.get("code","")), id=i+1, dt=row["date"].to_pydatetime(),
                    freq=freq, open=row["open"], close=row["close"], high=row["high"], low=row["low"],
                    vol=row.get(vol_col, row.get("volume",0)), amount=row.get("amount",0)))
    return bars

def compute_signals(code, freq_label='日线', signals_config=None):
    cfg = FREQ_CONFIG[freq_label]; sig_dir = cfg['signals_dir']; sig_dir.mkdir(parents=True, exist_ok=True)
    # 跳过已有信号文件的股票
    if (sig_dir / f"{code}.parquet").exists():
        return pd.DataFrame()  # 返回空 DataFrame，compute_all 会跳过
    df = _load_data(code, cfg['data_dir'])
    if df is None or df.empty: return None
    if len(df) < cfg['min_bars']: return None
    try: bars = _df_to_bars(df, cfg['freq'], cfg['vol_col'])
    except Exception as e: logger.warning("%s: bar convert fail: %s", code, e); return None
    if signals_config is None: signals_config = get_config(freq=freq_label)
    try: sigs_df = generate_czsc_signals(bars, signals_config=signals_config, sdt=SDT, init_n=min(cfg['init_n'], len(bars)), df=True)
    except Exception as e: logger.warning("%s: signal gen fail: %s", code, str(e)[:80]); return None
    if sigs_df is None or sigs_df.empty: return None
    sigs_df = sigs_df.drop(columns=[c for c in ["freq","cache"] if c in sigs_df.columns])
    out_path = sig_dir / f"{code}.parquet"; sigs_df.to_parquet(out_path, index=False)
    return sigs_df

def compute_all(codes, freq_label='日线', signals_config=None):
    total = len(codes); success = 0; t0 = time.time()
    for i, code in enumerate(codes):
        sigs_df = compute_signals(code, freq_label, signals_config)
        if sigs_df is not None and not sigs_df.empty: success += 1
        if (i+1)%500==0:
            elapsed = time.time()-t0; rate = (i+1)/elapsed; eta = (total-i-1)/rate if rate>0 else 0
            logger.info("进度: %d/%d, 成功 %d, 速率 %.1f/s, ETA %.0fs", i+1, total, success, rate, eta)
    elapsed = time.time()-t0
    logger.info("%s信号计算完成: %d/%d 成功, 耗时 %.0fs", freq_label, success, total, elapsed)
    return success

def main():
    import sys
    freq = sys.argv[1] if len(sys.argv)>1 else '日线'
    if freq not in FREQ_CONFIG: logger.error("不支持: %s, 可选: %s", freq, list(FREQ_CONFIG.keys())); return
    cfg = FREQ_CONFIG[freq]; codes = sorted(p.stem for p in cfg['data_dir'].glob("*.parquet"))
    if not codes: logger.error("%s 目录为空", cfg['data_dir']); return
    config = get_config(freq=freq)
    logger.info("%s: 配置 %d 个, 待计算 %d 只", freq, len(config), len(codes))
    compute_all(codes, freq_label=freq, signals_config=config)

if __name__ == "__main__": main()
