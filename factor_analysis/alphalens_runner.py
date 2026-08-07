"""Alphalens 分析运行器 — 接收因子 DataFrame，生成 IC 分析 / 分层收益 / 换手率报告。"""
import warnings
from pathlib import Path
import alphalens as al
import numpy as np
import pandas as pd
from factor_analysis.factor_extractor import _build_prices_for_alphalens


def _pick_factor_columns(factor_df):
    skip = {"buy_type", "total_score"}
    return [c for c in factor_df.columns if c not in skip
            and factor_df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]


def run_alphalens_analysis(factor_df, forward_returns=None,
                           periods=(1, 5, 20), quantiles=5, output_dir=None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if factor_df.empty:
        return {"error": "因子数据为空"}

    factor_cols = _pick_factor_columns(factor_df)
    print(f"\n{'='*60}")
    print(f"Alphalens 因子分析 — {len(factor_cols)} 个因子, {len(periods)} 个周期")
    n_samples = len(factor_df)
    n_stocks = factor_df.index.get_level_values("asset").nunique()
    print(f"样本量: {n_samples} 条, 股票数: {n_stocks}")
    print(f"{'='*60}")

    all_summaries = {}
    for factor_name in factor_cols:
        # 单因子切片
        single_factor = factor_df[[factor_name]].copy()
        valid = single_factor[factor_name].notna()
        single_factor = single_factor[valid]
        if single_factor.empty:
            print(f"\n  [{factor_name}] 跳过：全部为 NaN")
            continue
        if len(single_factor) < 30:
            print(f"\n  [{factor_name}] 跳过：有效样本 < 30")
            continue

        try:
            prices_df = _build_prices_for_alphalens(single_factor)
            if prices_df is None or prices_df.shape[1] < 3:
                print(f"\n  [{factor_name}] 跳过：价格数据不足")
                continue

            factor_data = al.utils.get_clean_factor_and_forward_returns(
                single_factor[factor_name], prices_df,
                quantiles=quantiles, periods=periods, max_loss=0.35,
            )
            ic = al.performance.factor_information_coefficient(factor_data)
            mean_ret_by_q = al.performance.mean_return_by_quantile(factor_data)
            turnover = al.performance.factor_rank_autocorrelation(factor_data)

            print(f"\n  [{factor_name}]")
            summary = {"n_obs": len(single_factor)}
            for p in periods:
                p_key = f"{p}D"
                if p_key in ic.columns:
                    ic_mean = ic[p_key].mean()
                    ic_ir = ic[p_key].mean() / (ic[p_key].std() + 1e-9)
                    summary[f"IC_mean_{p_key}"] = round(float(ic_mean), 4)
                    summary[f"IC_IR_{p_key}"] = round(float(ic_ir), 2)
                    if p_key in mean_ret_by_q.columns:
                        top_q = mean_ret_by_q[p_key].iloc[-1]
                        bot_q = mean_ret_by_q[p_key].iloc[0]
                        spread = top_q - bot_q
                        summary[f"spread_{p_key}_bps"] = round(float(spread * 10000), 1)
                    print(f"    {p_key}: IC={summary[f'IC_mean_{p_key}']:.4f}, "
                          f"IR={summary[f'IC_IR_{p_key}']:.2f}, "
                          f"spread={summary.get(f'spread_{p_key}_bps', 'N/A')}bps")
            all_summaries[factor_name] = summary

            report_df = pd.DataFrame({
                "period": [f"{p}D" for p in periods],
                "IC_mean": [summary.get(f"IC_mean_{p}D", np.nan) for p in periods],
                "IC_IR": [summary.get(f"IC_IR_{p}D", np.nan) for p in periods],
                "spread_bps": [summary.get(f"spread_{p}D_bps", np.nan) for p in periods],
            })
            safe_name = factor_name.replace("/", "_").replace(" ", "_")
            report_df.to_csv(output_dir / f"{safe_name}_ic.csv", index=False)
        except Exception as e:
            print(f"\n  [{factor_name}] 分析失败: {e}")
            all_summaries[factor_name] = {"error": str(e)}

    print(f"\n{'='*60}")
    print(f"分析完成，报告保存至: {output_dir}")
    if all_summaries:
        summary_rows = []
        for name, s in all_summaries.items():
            row = {"factor": name}
            row.update(s)
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_dir / "_summary.csv", index=False)
        print(f"汇总表: {output_dir / '_summary.csv'}")
    return all_summaries


def quick_factor_ranking(summaries, period="5D"):
    ic_key = f"IC_mean_{period}"
    ranked = [(name, s[ic_key]) for name, s in summaries.items() if ic_key in s]
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  因子按 |IC({period})| 排序:")
    for i, (name, ic) in enumerate(ranked):
        bar = "█" * min(int(abs(ic) * 50), 30)
        print(f"  {i+1:2d}. {name:<20s}  IC={ic:+.4f}  {bar}")
    return ranked
