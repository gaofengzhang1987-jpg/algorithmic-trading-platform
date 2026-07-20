"""统一信号解析器 — 项目中唯一版本的 parse_signal。"""
import pandas as pd


def parse_signal(val):
    """解析 CZSC 信号列的原始值。"""
    if pd.isna(val) or str(val) == "0":
        return {"v1": "", "v2": "", "v3": "", "score": "0"}
    parts = str(val).rsplit("_", 3)
    return {"v1": parts[0] if len(parts) >= 4 else "",
            "v2": parts[1] if len(parts) >= 4 else "",
            "v3": parts[2] if len(parts) >= 4 else "",
            "score": parts[3] if len(parts) >= 4 else "0"}
