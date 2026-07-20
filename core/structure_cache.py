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
    Returns (bi_low, gg_high) 或 (0, 0)。
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
