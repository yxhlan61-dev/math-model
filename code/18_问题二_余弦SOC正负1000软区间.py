"""问题二消融：余弦日末 SOC 基准、无风险微调、正负 1000 kWh 双侧软区间。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from problem2_solver_common import (
    CHECK_TOL,
    ETA_D,
    N,
    OFFICIAL_END,
    OFFICIAL_START,
    RHO,
    SOC_INITIAL,
    causal_forecasts,
    generate_scenarios,
    load_problem2_data,
    prediction_metrics,
    solve_daily_milp,
    validate_daily_solution,
)


BASE_SOC = 7000.0
AMPLITUDE_SOC = 800.0
PHASE_DAY = 15.0
BAND_HALF_WIDTH = 1000.0
RISK_WINDOW_DAYS = 56
RISK_MIN_HISTORY_DAYS = 7
RISK_GAIN_KWH = 200.0
RISK_ADJUSTMENT_CAP_KWH = 600.0


def seasonal_soc_reference(date: pd.Timestamp, base_soc: float) -> float:
    """Calendar-only SOC baseline; load/PV risk adjustment is applied separately."""
    return float(
        base_soc
        + AMPLITUDE_SOC * np.cos(2.0 * np.pi * (date.dayofyear - PHASE_DAY) / 365.0)
    )


def build_risk_audit(
    data, load_forecast: np.ndarray, pv_forecast: np.ndarray,
) -> pd.DataFrame:
    """Build per-date risk scores using only the scenario set available at each 0:00."""
    rows: list[dict] = []
    for d in range(1, len(data.dates)):
        load_s, pv_s, probabilities, _ = generate_scenarios(data, d, load_forecast, pv_forecast)
        net_energy = (load_s - pv_s).sum(axis=1)
        mean_net = float(probabilities @ net_energy)
        q80_net = float(np.quantile(net_energy, 0.8))
        rows.append({
            "day_index": d,
            "date": pd.Timestamp(data.dates[d]),
            "risk_mean_net_kwh": mean_net,
            "risk_q80_net_kwh": q80_net,
            "risk_score_kwh": q80_net - mean_net,
        })
    return pd.DataFrame(rows).set_index("day_index", drop=False)


def risk_adjustment(
    day_index: int, risk_audit: pd.DataFrame, enabled: bool,
) -> dict:
    current = risk_audit.loc[day_index]
    history = risk_audit.loc[
        (risk_audit["day_index"] >= max(1, day_index - RISK_WINDOW_DAYS))
        & (risk_audit["day_index"] < day_index)
    ]
    result = {
        "risk_score_kwh": float(current["risk_score_kwh"]),
        "risk_mean_net_kwh": float(current["risk_mean_net_kwh"]),
        "risk_q80_net_kwh": float(current["risk_q80_net_kwh"]),
        "risk_history_count": int(len(history)),
        "risk_history_start": None if history.empty else str(history.iloc[0]["date"].date()),
        "risk_history_end": None if history.empty else str(history.iloc[-1]["date"].date()),
        "risk_median_kwh": float("nan"),
        "risk_mad_scale_kwh": float("nan"),
        "daily_adjustment_kwh": 0.0,
    }
    if not enabled or len(history) < RISK_MIN_HISTORY_DAYS:
        return result
    values = history["risk_score_kwh"].to_numpy(dtype=float)
    median = float(np.median(values))
    scale = max(float(1.4826 * np.median(np.abs(values - median))), 100.0)
    adjustment = float(np.clip(
        RISK_GAIN_KWH * (result["risk_score_kwh"] - median) / scale,
        -RISK_ADJUSTMENT_CAP_KWH,
        RISK_ADJUSTMENT_CAP_KWH,
    ))
    result["risk_median_kwh"] = median
    result["risk_mad_scale_kwh"] = scale
    result["daily_adjustment_kwh"] = adjustment
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-soc", type=float, default=BASE_SOC)
    parser.add_argument("--risk-adjustment", action="store_true")
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()
    data = load_problem2_data()
    load_forecast, pv_forecast, forecast_audit = causal_forecasts(data)
    risk_audit = build_risk_audit(data, load_forecast, pv_forecast)
    penalty = float(ETA_D * np.quantile(data.price, 0.90))
    first_official_index = int(np.flatnonzero(data.dates == OFFICIAL_START)[0])
    initial_soc = SOC_INITIAL
    detail_rows: list[dict] = []
    daily_rows: list[dict] = []
    maxima = {
        "max_abs_balance_residual_kwh": 0.0,
        "max_abs_soc_residual_kwh": 0.0,
        "max_interday_soc_link_residual_kwh": 0.0,
        "max_simultaneous_charge_discharge_kwh": 0.0,
    }

    for d in range(first_official_index, len(data.dates)):
        date = pd.Timestamp(data.dates[d])
        seasonal_reference = seasonal_soc_reference(date, args.base_soc)
        risk = risk_adjustment(d, risk_audit, args.risk_adjustment)
        reference = float(np.clip(seasonal_reference + risk["daily_adjustment_kwh"], 5200.0, 8600.0))
        lower_target = reference - BAND_HALF_WIDTH
        upper_target = reference + BAND_HALF_WIDTH
        load_scenarios, pv_scenarios, probabilities, _ = generate_scenarios(
            data, d, load_forecast, pv_forecast
        )
        solution = solve_daily_milp(
            data.price,
            load_scenarios,
            pv_scenarios,
            probabilities,
            initial_soc,
            terminal_soc=None,
            reserve_target=lower_target,
            reserve_penalty=penalty,
            terminal_upper_target=upper_target,
            terminal_upper_penalty=penalty,
        )
        validation = validate_daily_solution(
            solution, data.price, load_scenarios, pv_scenarios, initial_soc, terminal_soc=None
        )
        if not validation["feasible"]:
            raise RuntimeError(f"{date.date()} daily feasibility check failed: {validation}")

        actual_emergency = np.maximum(
            data.load_kwh[d] + solution.charge - solution.purchase - data.pv_kwh[d] - solution.discharge,
            0.0,
        )
        actual_surplus = np.maximum(
            solution.purchase + data.pv_kwh[d] + solution.discharge - data.load_kwh[d] - solution.charge,
            0.0,
        )
        actual_balance = (
            solution.purchase + actual_emergency + data.pv_kwh[d] - actual_surplus
            + solution.discharge - data.load_kwh[d] - solution.charge
        )
        prior = np.r_[initial_soc, solution.soc[:-1]]
        soc_residual = solution.soc - prior - 0.9 * solution.charge + solution.discharge / 0.9
        normal_cost = float(data.price @ solution.purchase)
        emergency_cost = float((5.0 * data.price) @ actual_emergency)
        end_soc = float(solution.soc[-1])
        maxima["max_abs_balance_residual_kwh"] = max(
            maxima["max_abs_balance_residual_kwh"], float(np.max(np.abs(actual_balance)))
        )
        maxima["max_abs_soc_residual_kwh"] = max(
            maxima["max_abs_soc_residual_kwh"], float(np.max(np.abs(soc_residual)))
        )
        maxima["max_simultaneous_charge_discharge_kwh"] = max(
            maxima["max_simultaneous_charge_discharge_kwh"],
            float(np.max(np.minimum(solution.charge, solution.discharge))),
        )

        daily_rows.append({
            "date": date,
            "initial_soc_kwh": initial_soc,
            "seasonal_reference_soc_kwh": seasonal_reference,
            "daily_adjustment_kwh": risk["daily_adjustment_kwh"],
            "final_reference_soc_kwh": reference,
            "band_lower_soc_kwh": lower_target,
            "band_upper_soc_kwh": upper_target,
            "end_soc_kwh": end_soc,
            "below_band_kwh": solution.reserve_shortfall_kwh,
            "above_band_kwh": solution.terminal_excess_kwh,
            "terminal_band_penalty_yuan": solution.reserve_penalty_yuan + solution.terminal_excess_penalty_yuan,
            "normal_cost_yuan": normal_cost,
            "emergency_cost_yuan": emergency_cost,
            "total_cost_yuan": normal_cost + emergency_cost,
            "emergency_kwh": float(actual_emergency.sum()),
            "surplus_kwh": float(actual_surplus.sum()),
            "charge_kwh": float(solution.charge.sum()),
            "discharge_kwh": float(solution.discharge.sum()),
            **risk,
        })
        for t in range(N):
            detail_rows.append({
                "date": date,
                "period": t + 1,
                "price_yuan_per_kwh": float(data.price[t]),
                "actual_load_kwh": float(data.load_kwh[d, t]),
                "forecast_load_kwh": float(load_forecast[d, t]),
                "actual_pv_kwh": float(data.pv_kwh[d, t]),
                "forecast_pv_kwh": float(pv_forecast[d, t]),
                "purchase_kwh": float(solution.purchase[t]),
                "charge_kwh": float(solution.charge[t]),
                "discharge_kwh": float(solution.discharge[t]),
                "start_soc_kwh": float(initial_soc if t == 0 else solution.soc[t - 1]),
                "end_soc_kwh": float(solution.soc[t]),
                "emergency_kwh": float(actual_emergency[t]),
                "surplus_kwh": float(actual_surplus[t]),
                "balance_residual_kwh": float(actual_balance[t]),
            })
        if d > first_official_index:
            maxima["max_interday_soc_link_residual_kwh"] = max(
                maxima["max_interday_soc_link_residual_kwh"],
                abs(initial_soc - float(daily_rows[-2]["end_soc_kwh"])),
            )
        initial_soc = end_soc
        if (d - first_official_index + 1) % 25 == 0 or d == len(data.dates) - 1:
            cumulative = pd.DataFrame(daily_rows)
            print(
                f"{date.date()} complete; cumulative actual cost="
                f"{cumulative['total_cost_yuan'].sum():.2f} yuan; end SOC={end_soc:.1f}",
                flush=True,
            )

    daily = pd.DataFrame(daily_rows)
    detail = pd.DataFrame(detail_rows)
    tag = args.output_tag or (
        f"problem2_cosine{args.base_soc:g}_band1000_"
        f"{'riskadjust' if args.risk_adjustment else 'noadjust'}"
    )
    out_dir = Path(__file__).resolve().parents[1] / "tmp" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(forecast_audit).to_csv(out_dir / "forecast_audit.csv", index=False, encoding="utf-8-sig")
    report = {
        "model": "cosine seasonal terminal SOC band ±1000 kWh"
        + (" with causal scenario-risk adjustment" if args.risk_adjustment else " without risk adjustment"),
        "official_interval": f"{OFFICIAL_START.date()} to {OFFICIAL_END.date()}",
        "seasonal_curve": {"center_kwh": args.base_soc, "amplitude_kwh": AMPLITUDE_SOC, "phase_day": PHASE_DAY},
        "band_half_width_kwh": BAND_HALF_WIDTH,
        "risk_adjustment": {
            "enabled": bool(args.risk_adjustment),
            "quantile": 0.8,
            "window_days": RISK_WINDOW_DAYS,
            "minimum_history_days": RISK_MIN_HISTORY_DAYS,
            "gain_kwh_per_robust_scale": RISK_GAIN_KWH,
            "cap_kwh": RISK_ADJUSTMENT_CAP_KWH,
            "mean_kwh": float(daily["daily_adjustment_kwh"].mean()),
            "std_kwh": float(daily["daily_adjustment_kwh"].std(ddof=0)),
            "positive_days": int((daily["daily_adjustment_kwh"] > CHECK_TOL).sum()),
            "negative_days": int((daily["daily_adjustment_kwh"] < -CHECK_TOL).sum()),
            "positive_cap_days": int((daily["daily_adjustment_kwh"] >= RISK_ADJUSTMENT_CAP_KWH - CHECK_TOL).sum()),
            "negative_cap_days": int((daily["daily_adjustment_kwh"] <= -RISK_ADJUSTMENT_CAP_KWH + CHECK_TOL).sum()),
        },
        "terminal_penalty_yuan_per_kwh": penalty,
        "normal_cost_yuan": float(daily["normal_cost_yuan"].sum()),
        "emergency_cost_yuan": float(daily["emergency_cost_yuan"].sum()),
        "total_cost_yuan": float(daily["total_cost_yuan"].sum()),
        "emergency_kwh": float(daily["emergency_kwh"].sum()),
        "surplus_kwh": float(daily["surplus_kwh"].sum()),
        "end_soc_mean_kwh": float(daily["end_soc_kwh"].mean()),
        "end_soc_std_kwh": float(daily["end_soc_kwh"].std(ddof=0)),
        "end_soc_min_kwh": float(daily["end_soc_kwh"].min()),
        "end_soc_max_kwh": float(daily["end_soc_kwh"].max()),
        "below_band_days": int((daily["below_band_kwh"] > CHECK_TOL).sum()),
        "above_band_days": int((daily["above_band_kwh"] > CHECK_TOL).sum()),
        "at_lower_band_days": int((np.abs(daily["end_soc_kwh"] - daily["band_lower_soc_kwh"]) <= CHECK_TOL).sum()),
        "at_upper_band_days": int((np.abs(daily["end_soc_kwh"] - daily["band_upper_soc_kwh"]) <= CHECK_TOL).sum()),
        "physical_checks": maxima,
        "prediction_metrics": {
            "load": prediction_metrics(data.load_kwh, load_forecast),
            "pv": prediction_metrics(data.pv_kwh, pv_forecast, pv=True),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
