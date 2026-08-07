"""数据转换器 — 现在已不再需要 Qlib 二进制格式。保留接口以兼容。

当前架构: 直接使用 lightgbm 训练，从 parquet 文件读取数据，无需 Qlib 数据格式。
"""
from pathlib import Path
from core.constants import DATA_DIR, BASE_DIR


def convert_all(codes=None, force_csv=False):
    """占位函数 — 当前架构不需要 Qlib 二进制转换。"""
    print("当前架构（直接 LightGBM）不需要 Qlib 数据转换。跳过。")
    print("直接运行 qlib_ml.train() 即可训练。")


def parquet_to_csv(*args, **kwargs):
    print("不再需要 CSV 转换。跳过。")


def csv_to_qlib_bin(*args, **kwargs):
    print("不再需要 Qlib 二进制。跳过。")
