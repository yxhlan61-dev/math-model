"""将已验证的余弦 SOC 风险微调结果写入问题二正式中间结果与工作簿数据包。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from problem2_solver_common import INTERMEDIATE_DIR, N, load_problem2_data


ROOT = Path(__file__).resolve().parents[1]
RISK_DIR = ROOT / "tmp" / "problem2_cosine7000_band1000_riskadjust"
NO_SOC_GLOB = "*无日备惩罚*逐日汇总*.csv"


def column(frame: pd.DataFrame, text: str) -> str:
    return next(name for name in frame.columns if text in str(name))


def merge_emergency(date: str, values: np.ndarray, labels: list[str]) -> list[dict]:
    result: list[dict] = []
    start: int | None = None
    for t in range(N + 1):
        active = t < N and values[t] > 1e-6
        if active and start is None:
            start = t
        if not active and start is not None:
            end = t - 1
            result.append({
                "date": date,
                "interval": f"{labels[start].split('-')[0]}-{labels[end].split('-')[1]}",
                "emergency_kwh": float(values[start:t].sum()),
            })
            start = None
    return result


def main() -> None:
    data = load_problem2_data()
    daily = pd.read_csv(RISK_DIR / "daily.csv", parse_dates=["date"])
    detail = pd.read_csv(RISK_DIR / "detail.csv", parse_dates=["date"])
    no_soc_path = next(INTERMEDIATE_DIR.glob(NO_SOC_GLOB))
    no_soc = pd.read_csv(no_soc_path, encoding="utf-8-sig")
    no_soc_date = column(no_soc, "日期")
    no_soc_cost = column(no_soc, "实际总费用")
    no_soc_end = column(no_soc, "日末储电量")
    no_soc[no_soc_date] = pd.to_datetime(no_soc[no_soc_date])
    no_soc = no_soc[no_soc[no_soc_date] >= daily["date"].min()].copy()

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(INTERMEDIATE_DIR / "问题二_余弦风险_逐日结果.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(INTERMEDIATE_DIR / "问题二_余弦风险_逐时结果.csv", index=False, encoding="utf-8-sig")

    no_soc_monthly = no_soc.groupby(no_soc[no_soc_date].dt.to_period("M"))[no_soc_cost].sum()
    current_monthly = daily.groupby(daily["date"].dt.to_period("M"))["total_cost_yuan"].sum()
    comparison_monthly = pd.DataFrame({
        "month": current_monthly.index.astype(str),
        "no_terminal_soc_cost_yuan": no_soc_monthly.reindex(current_monthly.index).to_numpy(),
        "cosine_risk_cost_yuan": current_monthly.to_numpy(),
    })
    comparison_monthly["difference_new_minus_no_constraint_yuan"] = (
        comparison_monthly["cosine_risk_cost_yuan"] - comparison_monthly["no_terminal_soc_cost_yuan"]
    )
    comparison_monthly.to_csv(INTERMEDIATE_DIR / "问题二_余弦风险_与无SOC终端约束月度对比.csv", index=False, encoding="utf-8-sig")

    emergency_rows: list[dict] = []
    purchase_rows: list[dict] = []
    storage_rows: list[dict] = []
    for _, row in daily.iterrows():
        date = pd.Timestamp(row["date"])
        day = detail[detail["date"] == date].sort_values("period")
        emergency_rows.extend(merge_emergency(date.strftime("%Y-%m-%d"), day["emergency_kwh"].to_numpy(), data.interval_labels))
        purchase_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "purchase": day["purchase_kwh"].astype(float).tolist(),
            "total_purchase": float(day["purchase_kwh"].sum()),
            "total_cost": float(row["total_cost_yuan"]),
        })
        storage_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "charge_4h": [float(day["charge_kwh"].iloc[i:i + 24].sum()) for i in range(0, N, 24)],
            "discharge_4h": [float(day["discharge_kwh"].iloc[i:i + 24].sum()) for i in range(0, N, 24)],
            "initial_soc": float(row["initial_soc_kwh"]),
            "terminal_soc": float(row["end_soc_kwh"]),
        })

    no_soc_total = float(no_soc[no_soc_cost].sum())
    no_soc_end_values = no_soc[no_soc_end].astype(float)
    summary = {
        "model_name": "下移余弦日末SOC基准与因果场景风险微调MILP",
        "official_interval": "2025-02-01至2025-12-31",
        "days": int(len(daily)),
        "normal_cost_yuan": float(daily["normal_cost_yuan"].sum()),
        "emergency_cost_yuan": float(daily["emergency_cost_yuan"].sum()),
        "total_cost_yuan": float(daily["total_cost_yuan"].sum()),
        "purchase_kwh": float(detail["purchase_kwh"].sum()),
        "emergency_kwh": float(daily["emergency_kwh"].sum()),
        "surplus_kwh": float(daily["surplus_kwh"].sum()),
        "charge_kwh": float(daily["charge_kwh"].sum()),
        "discharge_kwh": float(daily["discharge_kwh"].sum()),
        "end_soc_mean_kwh": float(daily["end_soc_kwh"].mean()),
        "end_soc_std_kwh": float(daily["end_soc_kwh"].std(ddof=0)),
        "end_soc_last_kwh": float(daily.iloc[-1]["end_soc_kwh"]),
        "risk_adjust_mean_kwh": float(daily["daily_adjustment_kwh"].mean()),
        "risk_adjust_std_kwh": float(daily["daily_adjustment_kwh"].std(ddof=0)),
        "risk_positive_days": int((daily["daily_adjustment_kwh"] > 1e-6).sum()),
        "risk_negative_days": int((daily["daily_adjustment_kwh"] < -1e-6).sum()),
        "risk_cap_days": int((daily["daily_adjustment_kwh"] >= 600.0 - 1e-6).sum()),
        "at_lower_band_days": int((np.abs(daily["end_soc_kwh"] - daily["band_lower_soc_kwh"]) <= 1e-6).sum()),
        "no_terminal_soc_total_yuan": no_soc_total,
        "difference_new_minus_no_terminal_soc_yuan": float(daily["total_cost_yuan"].sum() - no_soc_total),
        "no_terminal_soc_end_mean_kwh": float(no_soc_end_values.mean()),
        "no_terminal_soc_end_last_kwh": float(no_soc_end_values.iloc[-1]),
    }
    bundle = {
        "summary": summary,
        "purchaseRows": purchase_rows,
        "storageRows": storage_rows,
        "emergencyRows": emergency_rows,
        "dailyRows": daily.assign(date=daily["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        "comparisonMonthly": comparison_monthly.to_dict("records"),
    }
    (INTERMEDIATE_DIR / "问题二_余弦风险_工作簿数据.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
