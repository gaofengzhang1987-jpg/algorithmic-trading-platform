"""季度重训练 —— 一键完成，自动对比旧模型并决定是否替换。

用法:
    python3 qlib_ml/quarterly_retrain.py           # 交互模式，确认后替换
    python3 qlib_ml/quarterly_retrain.py --dry-run # 仅训练+对比，不替换
    python3 qlib_ml/quarterly_retrain.py --force   # 不管对比结果，强制替换
"""
import argparse, json, shutil, sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qlib_ml.trainer import train


MODEL_DIR = Path(__file__).resolve().parent / "models"
ACTIVE_MODEL = MODEL_DIR / "lgb_model.txt"
ACTIVE_CONFIG = MODEL_DIR / "lgb_model_config.json"
BACKUP_DIR = MODEL_DIR / "backups"


def _latest_data_date():
    """扫描 data/daily/ 下 parquet，返回最新日期。"""
    from core.constants import DATA_DIR
    dates = []
    for f in Path(DATA_DIR).glob("*.parquet"):
        df = pd.read_parquet(f, columns=["date"])
        dates.append(pd.Timestamp(df["date"].max()))
    return max(dates).to_pydatetime() if dates else datetime.now()


def _compute_periods():
    """根据实际数据日期计算训练/验证区间。

    训练: 2020-01-01 → 最新数据前推 3 个月
    验证: 训练终点 → 最新数据日期（不是当前日期，避免数据还没到位就训练）
    """
    data_end = _latest_data_date()
    train_end = data_end - timedelta(days=90)
    train_period = ("2020-01-01", train_end.strftime("%Y-%m-%d"))
    valid_period = (train_end.strftime("%Y-%m-%d"), data_end.strftime("%Y-%m-%d"))
    return train_period, valid_period


def _backup_active():
    """备份当前活跃模型到 backups/。"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if ACTIVE_MODEL.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_model = BACKUP_DIR / f"lgb_model_{ts}.txt"
        dst_config = BACKUP_DIR / f"lgb_model_{ts}_config.json"
        shutil.copy2(ACTIVE_MODEL, dst_model)
        if ACTIVE_CONFIG.exists():
            shutil.copy2(ACTIVE_CONFIG, dst_config)
        return dst_model, dst_config
    return None, None


def _load_old_corr():
    """从旧模型 config 读取验证集 corr，若无则返回 None。"""
    if not ACTIVE_CONFIG.exists():
        return None
    cfg = json.loads(ACTIVE_CONFIG.read_text())
    return cfg.get("corr")


def main():
    parser = argparse.ArgumentParser(description="季度重训练 LGB 模型")
    parser.add_argument("--dry-run", action="store_true", help="仅训练对比，不替换")
    parser.add_argument("--force", action="store_true", help="强制替换，跳过对比")
    parser.add_argument("--max-stocks", type=int, default=None, help="训练股票数上限（默认全量）")
    args = parser.parse_args()

    train_period, valid_period = _compute_periods()

    print("=" * 60)
    print("季度重训练 — LightGBM 45 维模型")
    print(f"  训练区间: {train_period[0]} → {train_period[1]}")
    print(f"  验证区间: {valid_period[0]} → {valid_period[1]}")
    print(f"  股票数量: {'全量' if args.max_stocks is None else args.max_stocks}")
    print("=" * 60)

    # ── 训练新模型 ──
    ts = datetime.now().strftime("%Y%m%d")
    model_name = f"lgb_model_q{ts}"
    model_path = train(
        train_period=train_period,
        valid_period=valid_period,
        horizon=20,
        model_name=model_name,
        max_stocks=args.max_stocks,
    )

    # ── 读取新模型 corr ──
    new_config_path = MODEL_DIR / f"{model_name}_config.json"
    new_config = json.loads(new_config_path.read_text())
    new_corr = new_config.get("corr")

    old_corr = _load_old_corr()

    # ── 对比 ──
    print()
    print("=" * 60)
    print("对比结果")
    print(f"  旧模型 corr: {old_corr if old_corr is not None else '未知（无 config）'}")
    print(f"  新模型 corr: {new_corr:.4f}" if new_corr is not None else "  新模型 corr: 未知")
    print(f"  新模型样本: {new_config.get('n_samples', '?')}")
    print("=" * 60)

    if args.dry_run:
        print("[DRY-RUN] 不替换模型，新模型保存在:", model_path)
        return

    # 安全阀：corr 为负 = 模型比随机还差，不管什么情况都不替换
    if new_corr is not None and new_corr < 0:
        print(f"⛔ 新模型 corr={new_corr:.4f} < 0（比随机还差），拒绝替换。")
        return

    if args.force:
        print("[FORCE] 强制替换...")
    elif old_corr is not None and new_corr is not None and new_corr <= old_corr:
        print(f"⚠️  新模型 corr ({new_corr:.4f}) ≤ 旧模型 corr ({old_corr:.4f})")
        print("   不替换。如需强制替换请用 --force")
        return
    elif old_corr is None:
        print("旧模型无 corr 记录，默认替换...")

    # ── 替换 ──
    backup_model, backup_config = _backup_active()
    if backup_model:
        print(f"  已备份旧模型: {backup_model}")

    shutil.copy2(MODEL_DIR / f"{model_name}.txt", ACTIVE_MODEL)
    shutil.copy2(new_config_path, ACTIVE_CONFIG)

    # 更新 config 记录 corr
    cfg = json.loads(ACTIVE_CONFIG.read_text())
    cfg["replaced_at"] = datetime.now().isoformat()
    cfg["replaced_from"] = str(backup_model) if backup_model else "none"
    ACTIVE_CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    print(f"✅ 模型已替换: {model_name}.txt → lgb_model.txt")


if __name__ == "__main__":
    main()
