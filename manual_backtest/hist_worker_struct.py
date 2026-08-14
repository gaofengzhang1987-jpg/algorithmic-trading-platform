#!/usr/bin/env python3
"""struct_cache 构建 worker — 从已过滤的 K 线数据提取笔和中枢。"""
import sys, json, time, math
from pathlib import Path
import pandas as pd
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from czsc import CZSC, RawBar, Freq
from czsc.objects import Direction

MIN_BARS = {'日线': 120, '周线': 60, '30分钟': 200}

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    codes = cfg['codes']
    data_dir = Path(cfg['data_dir'])
    struct_dir = Path(cfg['struct_dir'])
    freq = cfg['freq']
    progress_file = Path(cfg['progress_file'])
    struct_dir.mkdir(parents=True, exist_ok=True)

    freq_enum = Freq.D if freq == '日线' else (Freq.F30 if freq == '30分钟' else Freq.W)
    total = len(codes)
    ok = fail = skip = 0
    t0 = time.time()

    for i, code in enumerate(codes):
        p = data_dir / f'{code}.parquet'
        out = struct_dir / f'{code}.parquet'
        if out.exists():
            skip += 1; continue
        if not p.exists():
            skip += 1; continue
        try:
            df = pd.read_parquet(p)
            dc = 'date' if 'date' in df.columns else 'dt'
            df[dc] = pd.to_datetime(df[dc])
            if len(df) < MIN_BARS[freq]:
                skip += 1; continue

            bars = [RawBar(symbol=code, id=j+1, dt=row[dc], freq=freq_enum,
                           open=float(row['open']), close=float(row['close']),
                           high=float(row['high']), low=float(row['low']),
                           vol=float(row.get('vol', 0) or row.get('volume', 0)),
                           amount=float(row.get('amount', 0)))
                    for j, (_, row) in enumerate(df.iterrows())]
            cz = CZSC(bars, max_bi_num=50)
        except Exception:
            fail += 1; continue

        try:
            rows = []
            for bi in cz.bi_list:
                d = "向下" if bi.direction == Direction.Down else "向上"
                rows.append({"direction": d,
                    "high": float(bi.high), "low": float(bi.low),
                    "sdt": bi.sdt, "edt": bi.edt, "power": float(bi.power),
                    "fx_b_mark": str(bi.fx_b.mark) if hasattr(bi.fx_b, "mark") else "",
                    "fx_b_low": float(bi.fx_b.low) if hasattr(bi.fx_b, "low") else 0.0,
                    "fx_a_mark": str(bi.fx_a.mark) if hasattr(bi.fx_a, "mark") else "",
                    "fx_b_has_zs": bool(getattr(bi.fx_b, "has_zs", False)),
                    "pivot_dir": math.nan, "pivot_gg": math.nan,
                    "pivot_zd": math.nan, "pivot_zg": math.nan})

            bl = cz.bi_list
            j = 0
            while j < len(bl) - 2:
                a, b, c = bl[j], bl[j+1], bl[j+2]
                oh = min(a.high, b.high, c.high)
                ol = max(a.low, b.low, c.low)
                if oh > ol:
                    zg, zd = oh, ol
                    gg, dd = oh, ol
                    k = j + 3
                    while k < len(bl):
                        nx = bl[k]
                        if nx.high >= zd and nx.low <= zg:
                            gg = max(gg, nx.high); dd = min(dd, nx.low)
                            k += 1
                        else: break
                    lv = bl[k] if k < len(bl) else None
                    if lv and lv.direction == Direction.Up: pd_ = "上涨"
                    elif lv and lv.direction == Direction.Down: pd_ = "下跌"
                    else: pd_ = "上涨" if bl[j].direction == Direction.Up else "下跌"
                    rows.append({"direction": "pivot",
                        "high": float(gg), "low": float(dd),
                        "sdt": bl[j].sdt, "edt": bl[k-1].edt if k < len(bl) else bl[-1].edt,
                        "power": 0.0, "fx_b_mark": "", "fx_b_low": 0.0, "fx_a_mark": "",
                        "fx_b_has_zs": pd.NA,
                        "pivot_dir": pd_, "pivot_gg": float(gg),
                        "pivot_zd": float(zd), "pivot_zg": float(zg)})
                    j = k
                else:
                    j += 1

            df_out = pd.DataFrame(rows)
            col_order = ["direction","edt","fx_a_mark","fx_b_low","fx_b_mark","fx_b_has_zs",
                         "high","low","pivot_dir","pivot_gg","pivot_zd","pivot_zg",
                         "power","sdt"]
            df_out = df_out[col_order]
            df_out.to_parquet(out, index=False)
            ok += 1
        except Exception:
            fail += 1

        if (i+1) % 50 == 0 or i == total-1:
            elapsed = time.time()-t0
            rate = (i+1)/elapsed if elapsed>0 else 0
            progress_file.write_text(f'{i+1}/{total} ok={ok} fail={fail} skip={skip} {rate:.1f}只/s')

    print(f'DONE ok={ok} fail={fail} skip={skip} {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
