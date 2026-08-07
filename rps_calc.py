#!/usr/bin/env python3
"""
RPS (Relative Price Strength) 计算脚本
- mode full:    第一次全量计算，生成全部历史数据 + daily_close.parquet
- mode refresh: 增量更新，从 daily_close.parquet（合并文件, 1 次 IO）读取新增数据

输出:
  data/reference/close_matrix.parquet   收盘价宽矩阵（日期 × 股票）
  data/reference/stock_rps.parquet      个股 RPS
  data/reference/industry_rps.parquet   板块 RPS
  data/daily_close.parquet              合并收盘价（date, code, close）

性能:
  full:     ~150s（读 4991 文件 + 计算 + 序列化）
  refresh:  ~10s （读 1 个 daily_close + incremental RPS）
  无 daily_close 时的 refresh: ~50s（扫描 4991 文件，仅首次 / daily 更新后）
"""

import os, sys, time, glob
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)  # RPS计算

# ─── 配置 ────────────────────────────────────────────────────────────────
BASE        = os.path.abspath(os.path.dirname(__file__))
DATA_DIR    = os.path.join(BASE, 'data/daily')
IND_CLS     = os.path.join(BASE, 'data/industry_classification.parquet')
REF_DIR     = os.path.join(BASE, 'data/reference')
DAILY_CLOSE = os.path.join(BASE, 'data/daily_close.parquet')   # 合并收盘价

CLOSE_MATRIX = os.path.join(REF_DIR, 'close_matrix.parquet')
STOCK_RPS    = os.path.join(REF_DIR, 'stock_rps.parquet')
INDUSTRY_RPS = os.path.join(REF_DIR, 'industry_rps.parquet')

PERIODS = [20, 60, 120, 250]

# ─── 工具 ─────────────────────────────────────────────────────────────────

def tic(): return time.time()

def toc(t0, label=""):
    print(f"  [{label}] {time.time() - t0:.1f}s"); sys.stdout.flush()

def load_industry_map():
    ind = pd.read_parquet(IND_CLS)
    return dict(zip(ind['code'], ind['name'])), dict(zip(ind['code'], ind['industry']))


def generate_daily_close_from_matrix(mat):
    """从 close_matrix (wide) 生成 daily_close.parquet (long: date, code, close)。"""
    print("  生成 daily_close.parquet..."); sys.stdout.flush()
    t0 = tic()
    long = mat.stack().reset_index()
    long.columns = ['date', 'code', 'close']
    long.sort_values(['date', 'code']).to_parquet(DAILY_CLOSE, index=False)
    toc(t0, f"daily_close 生成完毕: {DAILY_CLOSE} ({len(long):,} 行)")


# ─── 核心计算 ──────────────────────────────────────────────────────────────

def build_close_matrix():
    """全量：读取所有 daily parquet → wide close matrix。"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    c2n, _ = load_industry_map(); valid = set(c2n.keys())
    print(f"  读取 {len(files)} 只股票日线..."); sys.stdout.flush()
    t0 = tic()
    chunks = []
    for f in files:
        code = os.path.basename(f).replace('.parquet', '')
        if code not in valid: continue
        df = pd.read_parquet(f, columns=['date', 'close']).set_index('date')
        chunks.append(df.rename(columns={'close': code})[[code]])
    mat = pd.concat(chunks, axis=1).astype(np.float32)
    toc(t0, "close matrix built")
    print(f"    shape: {mat.shape}")
    return mat


def compute_composite(rps_dict):
    """合成 RPS：120×0.5 + 250×0.3 + 60×0.2。仅当至少一个 RPS 有效时计算。"""
    has = rps_dict[120].notna() | rps_dict[250].notna() | rps_dict[60].notna()
    c = (rps_dict[120].astype(float).fillna(0) * 0.5
         + rps_dict[250].astype(float).fillna(0) * 0.3
         + rps_dict[60].astype(float).fillna(0) * 0.2)
    c = c.round().clip(1, 99); c[~has] = pd.NA
    return c.astype('Int64')


def compute_rps_single(mat, periods=None):
    """计算 RPS + 转 long format。periods 指定返回哪些日期。"""
    print("  计算 RPS..."); sys.stdout.flush()
    t0 = tic()
    rps_dict, ret_dict = {}, {}
    for p in PERIODS:
        ret = mat.pct_change(p) * 100
        rps = (ret.rank(axis=1, pct=True) * 99).round().astype('Int64').clip(upper=99)
        rps_dict[p] = rps; ret_dict[p] = ret
    toc(t0, "rps done")

    comp = compute_composite(rps_dict)
    sub_dates = mat.index.intersection(periods) if periods is not None else mat.index
    print(f"  序列化 {len(sub_dates)} 天数据..."); sys.stdout.flush()
    t1 = tic()
    base = mat.loc[sub_dates].stack().reset_index()
    base.columns = ['date', 'code', 'close']
    for p in PERIODS:
        base[f'rps_{p}d'] = rps_dict[p].loc[sub_dates].stack().values
        base[f'ret_{p}d'] = ret_dict[p].loc[sub_dates].stack().values
    base['rps_composite'] = comp.loc[sub_dates].stack().values
    c2n, _ = load_industry_map(); base['name'] = base['code'].map(c2n)
    pool = rps_dict[20].loc[sub_dates].notna().sum(axis=1)
    base['pool_size'] = pool.reindex(base['date']).values
    base = base.sort_values(['date', 'code']).reset_index(drop=True)
    cols = ['date', 'code', 'name',
            'ret_20d','ret_60d','ret_120d','ret_250d',
            'rps_20d','rps_60d','rps_120d','rps_250d','rps_composite','pool_size']
    toc(t1, "long format done")
    return base[cols], rps_dict, ret_dict, comp


def compute_industry_rps(stock_rps):
    print("  计算行业 RPS..."); sys.stdout.flush()
    t0 = tic()
    _, c2ind = load_industry_map()
    sr = stock_rps.copy(); sr['industry'] = sr['code'].map(c2ind)
    ind = sr.groupby(['date', 'industry']).agg(
        stock_count=('code', 'count'),
        rps_20d=('rps_20d', 'median'), rps_60d=('rps_60d', 'median'),
        rps_120d=('rps_120d', 'median'), rps_250d=('rps_250d', 'median'),
        rps_composite=('rps_composite', 'median'),
    ).reset_index()
    strong = sr[sr['rps_composite'] >= 90].groupby(['date','industry']).size().reset_index(name='top_10_pct_count')
    ind = ind.merge(strong, on=['date','industry'], how='left')
    ind['top_10_pct_count'] = ind['top_10_pct_count'].fillna(0).astype(int)
    ind = ind.sort_values(['date','industry']).reset_index(drop=True)
    toc(t0, "industry rps done")
    return ind


# ─── 增量：从 daily_close 读取新数据（快速路径） ─────────────────────────

def read_new_closes_from_daily_close(last_date):
    """从 daily_close.parquet 读取 >last_date 的数据，pivot 成 wide。"""
    dc = pd.read_parquet(DAILY_CLOSE, columns=['date', 'code', 'close'])
    new = dc[dc['date'] > last_date]
    if len(new) == 0:
        return None
    mat = new.pivot_table(index='date', columns='code', values='close').astype(np.float32)
    return mat


def append_new_closes(mat, new_mat):
    """将 new_mat 追加到 mat，去重并排序。"""
    m = pd.concat([mat, new_mat])
    m = m[~m.index.duplicated(keep='last')].sort_index()
    return m


# ─── 主流程 ───────────────────────────────────────────────────────────────

def mode_full():
    os.makedirs(REF_DIR, exist_ok=True)
    c2n, _ = load_industry_map()
    print(f"有效股票数: {len(c2n)}"); sys.stdout.flush()
    mat = build_close_matrix()
    print("  保存 close_matrix..."); mat.to_parquet(CLOSE_MATRIX); sys.stdout.flush()
    stock_rps, _, _, _ = compute_rps_single(mat, periods=None)
    print("  保存 stock_rps..."); stock_rps.to_parquet(STOCK_RPS, index=False); sys.stdout.flush()
    ind_rps = compute_industry_rps(stock_rps)
    print("  保存 industry_rps..."); ind_rps.to_parquet(INDUSTRY_RPS, index=False); sys.stdout.flush()
    # 同步生成 daily_close.parquet
    generate_daily_close_from_matrix(mat)
    print(f"  完成（{len(stock_rps):,} 行 stock | {len(ind_rps):,} 行 industry）")


def mode_refresh():
    if not os.path.exists(CLOSE_MATRIX):
        print("close_matrix 不存在，切换 full 模式。"); return mode_full()

    print("加载 close_matrix..."); sys.stdout.flush()
    mat = pd.read_parquet(CLOSE_MATRIX)
    last_date = mat.index.max()
    print(f"  当前最后日期: {last_date.date()}")
    c2n, _ = load_industry_map()

    # ── 尝试从 daily_close.parquet 快速读取 ──
    if os.path.exists(DAILY_CLOSE):
        print("从 daily_close.parquet 读取新数据..."); sys.stdout.flush()
        t0 = tic()
        new_mat = read_new_closes_from_daily_close(last_date)
        if new_mat is not None:
            mat = append_new_closes(mat, new_mat)
            mat.to_parquet(CLOSE_MATRIX)
            toc(t0, f"close_matrix 快速更新 (+{len(new_mat)} 天)")
        else:
            print("  无新数据。"); return
    else:
        # ── 慢速路径：扫描 4991 个 daily 文件并生成 daily_close ──
        print("daily_close.parquet 不存在，扫描 4991 文件（仅首次 / 每日数据更新后）...")
        sys.stdout.flush()
        t0 = tic()
        files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
        valid = set(c2n.keys())
        new_rows = []
        for f in files:
            code = os.path.basename(f).replace('.parquet', '')
            if code not in valid: continue
            df = pd.read_parquet(f, columns=['date', 'close'])
            sub = df[df['date'] > last_date]
            if len(sub):
                new_rows.append(sub.set_index('date').rename(columns={'close': code})[[code]])

        if not new_rows:
            # 扫描了但没有新数据 → 生成 daily_close（一次性）
            generate_daily_close_from_matrix(mat)
            print("无新数据。"); return

        new_mat = pd.concat(new_rows, axis=1).astype(np.float32)
        mat = append_new_closes(mat, new_mat)
        mat.to_parquet(CLOSE_MATRIX)
        toc(t0, f"close_matrix 慢速更新 (+{len(new_mat)} 天)")

        # 生成/更新 daily_close.parquet
        generate_daily_close_from_matrix(mat)

    # ── 计算新日期的 RPS ──
    new_dates = mat.index[mat.index > last_date]
    print(f"新增 {len(new_dates)} 天，计算 RPS..."); sys.stdout.flush()
    stock_new, _, _, _ = compute_rps_single(mat, periods=new_dates)

    if os.path.exists(STOCK_RPS):
        old = pd.read_parquet(STOCK_RPS)
        old = old[old['date'] < new_dates.min()]
        stock_new = pd.concat([old, stock_new], ignore_index=True)
    stock_new.to_parquet(STOCK_RPS, index=False)
    print(f"  stock_rps: {len(stock_new):,} 行"); sys.stdout.flush()

    ind_new = compute_industry_rps(stock_new[stock_new['date'].isin(new_dates)])
    if os.path.exists(INDUSTRY_RPS):
        old_ind = pd.read_parquet(INDUSTRY_RPS)
        old_ind = old_ind[old_ind['date'] < new_dates.min()]
        ind_new = pd.concat([old_ind, ind_new], ignore_index=True)
    ind_new.to_parquet(INDUSTRY_RPS, index=False)
    print(f"  industry_rps: {len(ind_new):,} 行"); sys.stdout.flush()
    print(f"增量完成。新日期: {new_dates.min().date()} ~ {new_dates.max().date()}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    if mode == 'full':    mode_full()
    elif mode == 'refresh': mode_refresh()
    else: print(f"用法: python3 rps_calc.py [full|refresh]"); sys.exit(1)
