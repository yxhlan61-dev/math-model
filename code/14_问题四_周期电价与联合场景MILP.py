"""问题四：周期电价预测、日内惯性更新、联合场景与问题4-2/4-3求解。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from problem2_solver_common import (
    CHECK_TOL, ETA_C, ETA_D, N, OFFICIAL_END, OFFICIAL_START, RESERVE_TARGET,
    SOC_INITIAL, causal_forecasts, generate_scenarios, load_problem2_data,
    merge_emergency_intervals, solve_daily_milp, validate_daily_solution, write_json,
)
from problem3_solver_common import (
    generate_horizon_scenarios, load_forecast_only, load_problem3_data,
    solve_horizon_milp, validate_horizon,
)
from problem4_price_common import (
    ISSUE_HOURS, PriceForecastData, build_price_forecasts, forecast_metrics,
    price_scenarios, reserve_penalty,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "code" / "outputs" / "问题四"
INTERMEDIATE_DIR = ROOT / "tmp" / "问题四_求解中间结果"
FIGURE_DIR = ROOT / "figures" / "问题四"
POLICIES = {
    "S0_仅0点": (),
    "S6_增加6点": (6,),
    "S612_增加6点12点": (6, 12),
    "S61218_全部四时点": (6, 12, 18),
}
OFFICIAL_POLICY = "S61218_全部四时点"
SCENARIO_DRAWS = 30
SOC_BASE = 7000.0
SOC_AMPLITUDE = 800.0
SOC_PHASE_DAY = 15.0
SOC_BAND_HALF_WIDTH = 1000.0
SOC_RISK_WINDOW = 56
SOC_RISK_MIN_HISTORY = 7
SOC_RISK_GAIN = 200.0
SOC_RISK_CAP = 600.0


def soc_reference(date: pd.Timestamp, risk_score: float, history: list[float]) -> dict:
    seasonal = SOC_BASE + SOC_AMPLITUDE * np.cos(
        2.0 * np.pi * (date.dayofyear - SOC_PHASE_DAY) / 365.0
    )
    hist = np.asarray(history[-SOC_RISK_WINDOW:], dtype=float)
    if len(hist) < SOC_RISK_MIN_HISTORY:
        median = float("nan"); scale = float("nan"); delta = 0.0
    else:
        median = float(np.median(hist))
        scale = float(max(1.4826 * np.median(np.abs(hist - median)), 100.0))
        delta = float(np.clip(SOC_RISK_GAIN * (risk_score - median) / scale, -SOC_RISK_CAP, SOC_RISK_CAP))
    reference = float(np.clip(seasonal + delta, 5200.0, 8600.0))
    return {"seasonal": float(seasonal), "risk_median": median, "risk_scale": scale,
            "risk_delta": delta, "reference": reference,
            "lower": reference - SOC_BAND_HALF_WIDTH, "upper": reference + SOC_BAND_HALF_WIDTH}


def seed_problem2_risk_history(data, load_forecast, pv_forecast) -> list[float]:
    history: list[float] = []
    for day_index in range(1, 31):
        load_s, pv_s, _, sampled_draws = generate_scenarios(data, day_index, load_forecast, pv_forecast)
        load_s, pv_s, probabilities, _ = _compress_problem2_scenarios(load_s, pv_s, sampled_draws)
        net = (load_s - pv_s).sum(axis=1)
        history.append(float(np.quantile(net, 0.8) - probabilities @ net))
    return history


def seed_problem3_risk_history(data, load_forecast) -> list[float]:
    history: list[float] = []
    for day_index in range(1, 31):
        load_s, pv_s, probabilities, _ = generate_horizon_scenarios(
            data, day_index, 0, 0, load_forecast, scenario_count=SCENARIO_DRAWS)
        net = (load_s - pv_s).sum(axis=1)
        history.append(float(np.quantile(net, 0.8) - probabilities @ net))
    return history


def _compress_problem2_scenarios(load_s, pv_s, sampled):
    if not sampled:
        return load_s, pv_s, np.ones(1), []
    sampled_array = np.asarray(sampled[:SCENARIO_DRAWS], dtype=int)
    load_s = load_s[:len(sampled_array)]
    pv_s = pv_s[:len(sampled_array)]
    unique, first, counts = np.unique(sampled_array, return_index=True, return_counts=True)
    return load_s[first], pv_s[first], counts / counts.sum(), unique.tolist()


def _maxima() -> dict:
    return {
        "max_abs_scenario_balance_residual_kwh": 0.0,
        "max_abs_soc_recursion_residual_kwh": 0.0,
        "max_abs_adjustment_residual_kwh": 0.0,
        "max_abs_actual_balance_residual_kwh": 0.0,
        "max_soc_link_residual_kwh": 0.0,
        "min_soc_kwh": float("inf"), "max_soc_kwh": -float("inf"),
        "max_charge_kwh": 0.0, "max_discharge_kwh": 0.0,
        "simultaneous_charge_discharge_count": 0,
        "simultaneous_up_down_count": 0, "max_mip_gap": 0.0,
    }


def _update_maxima(maxima: dict, check: dict) -> None:
    for key in ["max_abs_scenario_balance_residual_kwh", "max_abs_soc_recursion_residual_kwh",
                "max_abs_adjustment_residual_kwh", "max_charge_kwh", "max_discharge_kwh"]:
        if key in check:
            maxima[key] = max(maxima[key], float(check[key]))
    maxima["min_soc_kwh"] = min(maxima["min_soc_kwh"], float(check["min_soc_kwh"]))
    maxima["max_soc_kwh"] = max(maxima["max_soc_kwh"], float(check["max_soc_kwh"]))
    maxima["simultaneous_charge_discharge_count"] += int(check["simultaneous_charge_discharge_count"])
    maxima["simultaneous_up_down_count"] += int(check.get("simultaneous_up_down_count", 0))
    maxima["max_mip_gap"] = max(maxima["max_mip_gap"], float(check.get("mip_gap") or 0.0))


def simulate_42(data, prices: PriceForecastData, load_forecast, pv_forecast,
                end_date: pd.Timestamp, oracle: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    detail_rows, daily_rows = [], []
    maxima = _maxima()
    initial_soc = SOC_INITIAL
    risk_history = seed_problem2_risk_history(data, load_forecast, pv_forecast)
    max_day = int(np.searchsorted(data.dates.values, np.datetime64(end_date), side="right"))
    for d in range(31, max_day):
        date = data.dates[d]
        load_s, pv_s, _, sampled_draws = generate_scenarios(data, d, load_forecast, pv_forecast)
        load_s, pv_s, probabilities, sampled = _compress_problem2_scenarios(load_s, pv_s, sampled_draws)
        scenario_net = (load_s - pv_s).sum(axis=1)
        risk_score = float(np.quantile(scenario_net, 0.8) - probabilities @ scenario_net)
        soc_rule = soc_reference(pd.Timestamp(date), risk_score, risk_history)
        risk_history.append(risk_score)
        if oracle:
            price_s = np.broadcast_to(prices.actual[d], load_s.shape)
        else:
            price_s = price_scenarios(prices, d, 0, 0, sampled)
        kappa = reserve_penalty(prices.actual, d, ETA_D)
        solution = solve_daily_milp(
            price_s, load_s, pv_s, probabilities, initial_soc, None,
            reserve_target=soc_rule["lower"], reserve_penalty=kappa,
            terminal_upper_target=soc_rule["upper"], terminal_upper_penalty=kappa,
        )
        check = validate_daily_solution(solution, price_s, load_s, pv_s, initial_soc, None)
        if not check["feasible"]:
            raise RuntimeError(f"{date.date()} 问题4-2约束失败：{check}")
        _update_maxima(maxima, check)
        actual_emergency = np.maximum(
            data.load_kwh[d] + solution.charge - solution.purchase - data.pv_kwh[d] - solution.discharge, 0.0
        )
        actual_surplus = np.maximum(
            solution.purchase + data.pv_kwh[d] + solution.discharge - data.load_kwh[d] - solution.charge, 0.0
        )
        actual_balance = (solution.purchase + actual_emergency + data.pv_kwh[d] - actual_surplus
                          + solution.discharge - data.load_kwh[d] - solution.charge)
        plan_cost = float(prices.actual[d] @ solution.purchase)
        emergency_cost = float((5.0 * prices.actual[d]) @ actual_emergency)
        prior_soc = initial_soc
        for t in range(N):
            detail_rows.append({
                "口径": "完全信息价格下界" if oracle else "现实周期预测", "日期": date,
                "时段编号": t + 1, "时段": data.interval_labels[t],
                "预测电价(元/kWh)": prices.actual[d, t] if oracle else prices.day_ahead[d, t],
                "实际电价(元/kWh)": prices.actual[d, t], "计划购电量(kWh)": solution.purchase[t],
                "充电量(kWh)": solution.charge[t], "放电量(kWh)": solution.discharge[t],
                "时段初储电量(kWh)": prior_soc if t == 0 else solution.soc[t - 1],
                "时段末储电量(kWh)": solution.soc[t], "实际紧急购电量(kWh)": actual_emergency[t],
                "实际富余电量(kWh)": actual_surplus[t], "实际平衡残差(kWh)": actual_balance[t],
            })
        daily_rows.append({
            "口径": "完全信息价格下界" if oracle else "现实周期预测", "日期": date,
            "日初储电量(kWh)": initial_soc, "日末储电量(kWh)": float(solution.soc[-1]),
            "计划购电量(kWh)": float(solution.purchase.sum()),
            "紧急购电量(kWh)": float(actual_emergency.sum()), "富余电量(kWh)": float(actual_surplus.sum()),
            "充电量(kWh)": float(solution.charge.sum()), "放电量(kWh)": float(solution.discharge.sum()),
            "正常购电费(元)": plan_cost, "紧急购电费(元)": emergency_cost,
            "实际总费用(元)": plan_cost + emergency_cost, "日备不足惩罚系数": kappa,
            "SOC季节基准(kWh)": soc_rule["seasonal"], "SOC风险分数(kWh)": risk_score,
            "SOC每日微调(kWh)": soc_rule["risk_delta"], "SOC软区间下沿(kWh)": soc_rule["lower"],
            "SOC软区间上沿(kWh)": soc_rule["upper"],
            "价格场景数": len(probabilities), "MIP相对间隙": solution.solver_details.get("mip_gap"),
        })
        maxima["max_abs_actual_balance_residual_kwh"] = max(
            maxima["max_abs_actual_balance_residual_kwh"], float(np.max(np.abs(actual_balance))))
        if len(daily_rows) > 1 and d > 30:
            maxima["max_soc_link_residual_kwh"] = max(
                maxima["max_soc_link_residual_kwh"], abs(initial_soc - daily_rows[-2]["日末储电量(kWh)"]))
        initial_soc = float(solution.soc[-1])
    maxima["solve_count"] = max(0, max_day - 31)
    return pd.DataFrame(detail_rows), pd.DataFrame(daily_rows), maxima


def simulate_43(data, prices: PriceForecastData, load_forecast, policy_name: str,
                updates: tuple[int, ...], end_date: pd.Timestamp,
                oracle: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = data.base
    initial_soc = SOC_INITIAL
    detail_rows, daily_rows = [], []
    maxima = _maxima()
    max_day = int(np.searchsorted(base.dates.values, np.datetime64(end_date), side="right"))
    solve_count = 0
    risk_history = seed_problem3_risk_history(data, load_forecast)
    for d in range(31, max_day):
        date = base.dates[d]
        kappa = reserve_penalty(prices.actual, d, ETA_D)
        load_s, pv_s, probabilities, sampled = generate_horizon_scenarios(
            data, d, 0, 0, load_forecast, scenario_count=SCENARIO_DRAWS)
        scenario_net = (load_s - pv_s).sum(axis=1)
        risk_score = float(np.quantile(scenario_net, 0.8) - probabilities @ scenario_net)
        soc_rule = soc_reference(pd.Timestamp(date), risk_score, risk_history)
        risk_history.append(risk_score)
        price_s = (np.broadcast_to(prices.actual[d], load_s.shape) if oracle
                   else price_scenarios(prices, d, 0, 0, sampled))
        plan = solve_horizon_milp(
            price_s, load_s, pv_s, probabilities, initial_soc, None,
            reserve_target=soc_rule["lower"], reserve_penalty=kappa,
            terminal_upper_target=soc_rule["upper"], terminal_upper_penalty=kappa,
        )
        solve_count += 1
        checks = [(plan, load_s, pv_s, initial_soc, None)]
        original = plan.purchase.copy()
        purchase, charge, discharge, soc = (plan.purchase.copy(), plan.charge.copy(),
                                             plan.discharge.copy(), plan.soc.copy())
        source_issue = np.zeros(N, dtype=int)
        for issue_hour in updates:
            start = issue_hour * 6
            rolling_initial = float(soc[start - 1])
            load_s, pv_s, probabilities, sampled = generate_horizon_scenarios(
                data, d, issue_hour, start, load_forecast, scenario_count=SCENARIO_DRAWS)
            price_s = (np.broadcast_to(prices.actual[d, start:], load_s.shape) if oracle
                       else price_scenarios(prices, d, issue_hour, start, sampled))
            rolling = solve_horizon_milp(
                price_s, load_s, pv_s, probabilities, rolling_initial, None,
                original_plan=original[start:], reserve_target=soc_rule["lower"], reserve_penalty=kappa,
                terminal_upper_target=soc_rule["upper"], terminal_upper_penalty=kappa,
            )
            solve_count += 1
            checks.append((rolling, load_s, pv_s, rolling_initial, original[start:]))
            purchase[start:], charge[start:], discharge[start:], soc[start:] = (
                rolling.purchase, rolling.charge, rolling.discharge, rolling.soc)
            source_issue[start:] = issue_hour
        for solution, ls, ps, horizon_initial, horizon_plan in checks:
            _update_maxima(maxima, validate_horizon(solution, ls, ps, horizon_initial, horizon_plan))
        actual_emergency = np.maximum(
            base.load_kwh[d] + charge - purchase - base.pv_kwh[d] - discharge, 0.0)
        actual_surplus = np.maximum(
            purchase + base.pv_kwh[d] + discharge - base.load_kwh[d] - charge, 0.0)
        actual_balance = purchase + actual_emergency + base.pv_kwh[d] - actual_surplus + discharge - base.load_kwh[d] - charge
        upward = np.maximum(purchase - original, 0.0)
        downward = np.maximum(original - purchase, 0.0)
        actual_price = prices.actual[d]
        plan_cost = float(actual_price @ original)
        upward_cost = float((1.5 * actual_price) @ upward)
        downward_credit = float((-0.5 * actual_price) @ downward)
        emergency_cost = float((5.0 * actual_price) @ actual_emergency)
        scheduled_cost = plan_cost + upward_cost + downward_credit
        for t in range(N):
            issue_index = ISSUE_HOURS.index(int(source_issue[t]))
            detail_rows.append({
                "口径": "完全信息价格下界" if oracle else "现实滚动预测", "策略": policy_name,
                "日期": date, "时段编号": t + 1, "时段": base.interval_labels[t],
                "采用价格预测时点": int(source_issue[t]),
                "预测电价(元/kWh)": actual_price[t] if oracle else prices.issue_forecast[d, issue_index, t],
                "实际电价(元/kWh)": actual_price[t], "计划购电量(kWh)": original[t],
                "调整后购电量(kWh)": purchase[t], "上调量(kWh)": upward[t], "下调量(kWh)": downward[t],
                "充电量(kWh)": charge[t], "放电量(kWh)": discharge[t],
                "时段初储电量(kWh)": initial_soc if t == 0 else soc[t - 1],
                "时段末储电量(kWh)": soc[t], "实际紧急购电量(kWh)": actual_emergency[t],
                "实际富余电量(kWh)": actual_surplus[t], "实际平衡残差(kWh)": actual_balance[t],
            })
        daily_rows.append({
            "口径": "完全信息价格下界" if oracle else "现实滚动预测", "策略": policy_name, "日期": date,
            "日初储电量(kWh)": initial_soc, "日末储电量(kWh)": float(soc[-1]),
            "计划购电量(kWh)": float(original.sum()), "调整后购电量(kWh)": float(purchase.sum()),
            "上调量(kWh)": float(upward.sum()), "下调量(kWh)": float(downward.sum()),
            "紧急购电量(kWh)": float(actual_emergency.sum()), "富余电量(kWh)": float(actual_surplus.sum()),
            "充电量(kWh)": float(charge.sum()), "放电量(kWh)": float(discharge.sum()),
            "计划购电费(元)": plan_cost, "上调费用(元)": upward_cost, "下调抵扣(元)": downward_credit,
            "计划与调整结算费(元)": scheduled_cost, "紧急购电费(元)": emergency_cost,
            "实际总费用(元)": scheduled_cost + emergency_cost,
            "SOC季节基准(kWh)": soc_rule["seasonal"], "SOC风险分数(kWh)": risk_score,
            "SOC每日微调(kWh)": soc_rule["risk_delta"], "SOC软区间下沿(kWh)": soc_rule["lower"],
            "SOC软区间上沿(kWh)": soc_rule["upper"],
            "6点调整量(kWh)": float(np.abs(purchase[36:] - original[36:]).sum()) if 6 in updates else 0.0,
            "12点调整量(kWh)": float(np.abs(purchase[72:] - original[72:]).sum()) if 12 in updates else 0.0,
            "18点调整量(kWh)": float(np.abs(purchase[108:] - original[108:]).sum()) if 18 in updates else 0.0,
        })
        maxima["max_abs_actual_balance_residual_kwh"] = max(
            maxima["max_abs_actual_balance_residual_kwh"], float(np.max(np.abs(actual_balance))))
        if len(daily_rows) > 1 and d > 30:
            maxima["max_soc_link_residual_kwh"] = max(
                maxima["max_soc_link_residual_kwh"], abs(initial_soc - daily_rows[-2]["日末储电量(kWh)"]))
        initial_soc = float(soc[-1])
    maxima["solve_count"] = solve_count
    return pd.DataFrame(detail_rows), pd.DataFrame(daily_rows), maxima


def summarize(daily: pd.DataFrame) -> dict:
    frame = daily[(daily["日期"] >= OFFICIAL_START) & (daily["日期"] <= OFFICIAL_END)]
    numeric = [column for column in frame.columns if column.endswith("(kWh)") or column.endswith("(元)")]
    result = {column: float(frame[column].sum()) for column in numeric if column not in ["日初储电量(kWh)", "日末储电量(kWh)"]}
    result.update({
        "正式天数": int(len(frame)), "平均日末储电量(kWh)": float(frame["日末储电量(kWh)"].mean()),
        "12月31日末储电量(kWh)": float(frame.iloc[-1]["日末储电量(kWh)"]),
        "发生紧急购电天数": int((frame["紧急购电量(kWh)"] > CHECK_TOL).sum()),
    })
    return result


def build_bundle(detail42, daily42, detail43, daily43, summaries, validations, metrics) -> dict:
    def rows_for_submission(detail, daily, adjusted=False):
        detail = detail[(detail["日期"] >= OFFICIAL_START) & (detail["日期"] <= OFFICIAL_END)]
        daily = daily[(daily["日期"] >= OFFICIAL_START) & (daily["日期"] <= OFFICIAL_END)]
        purchase_rows, adjustment_rows, storage_rows, emergency_rows = [], [], [], []
        for date, group in detail.groupby("日期", sort=True):
            group = group.sort_values("时段编号")
            day = daily[daily["日期"] == date].iloc[0]
            purchase_rows.append({"date": date.strftime("%Y-%m-%d"), "values": group["计划购电量(kWh)"].tolist(),
                                  "total": day["计划购电量(kWh)"], "cost": day["计划购电费(元)"] if adjusted else day["正常购电费(元)"]})
            if adjusted:
                adjustment_rows.append({"date": date.strftime("%Y-%m-%d"), "values": group["调整后购电量(kWh)"].tolist(),
                                        "total": day["调整后购电量(kWh)"], "cost": day["计划与调整结算费(元)"]})
            storage_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "charge_4h": [float(group.iloc[i:i + 24]["充电量(kWh)"].sum()) for i in range(0, N, 24)],
                "discharge_4h": [float(group.iloc[i:i + 24]["放电量(kWh)"].sum()) for i in range(0, N, 24)],
                "initial_soc": float(day["日初储电量(kWh)"]), "terminal_soc": float(day["日末储电量(kWh)"]),
            })
            emergency_rows.extend(merge_emergency_intervals(date, group["实际紧急购电量(kWh)"].to_numpy(), group["时段"].tolist()))
        return {"purchaseRows": purchase_rows, "adjustmentRows": adjustment_rows,
                "storageRows": storage_rows, "emergencyRows": emergency_rows}
    return {
        "result42": rows_for_submission(detail42, daily42),
        "result43": rows_for_submission(detail43, daily43, adjusted=True),
        "summaries": summaries, "validations": validations, "forecastMetrics": metrics,
        "daily42": [{k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v) for k, v in row.items()} for row in daily42.to_dict("records")],
        "daily43": [{k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v) for k, v in row.items()} for row in daily43.to_dict("records")],
    }


def draw_figures(prices: PriceForecastData, detail43: pd.DataFrame, summaries: dict, metrics: dict) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 9})
    paths = []
    hour_ticks = np.arange(0, 25, 3)
    mean_grid = np.vstack([prices.actual[prices.dates.weekday == weekday].mean(axis=0) for weekday in range(7)])
    fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
    image = ax.imshow(mean_grid, aspect="auto", cmap="YlOrRd", extent=[0, 24, 6.5, -0.5])
    ax.set_yticks(range(7)); ax.set_yticklabels(["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
    ax.set_xlim(0, 24); ax.set_xticks(hour_ticks)
    ax.set_xlabel("时刻/h"); ax.set_ylabel("星期"); fig.colorbar(image, ax=ax, label="平均电价/(元/kWh)")
    path = FIGURE_DIR / "问题四_星期时刻平均电价热力图.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    # 与第二问“星期一至星期日平均负荷与光伏曲线”保持相同的时刻轴和配色。
    x = (np.arange(N) + 1.5) / 6.0
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_colors = ["#2166AC", "#4393C3", "#67A9CF", "#1B9E77", "#F28E2B", "#D62728", "#00A087"]
    for weekday, label in enumerate(weekday_names):
        is_friday_or_saturday = weekday in (4, 5)
        ax.plot(
            x,
            mean_grid[weekday],
            color=weekday_colors[weekday],
            linewidth=1.8 if is_friday_or_saturday else 1.15,
            linestyle="--" if is_friday_or_saturday else "-",
            label=label,
        )
    ax.set_xlabel("时刻/h"); ax.set_ylabel("平均电价/(元/kWh)")
    ax.set_xlim(0, 24); ax.set_xticks(hour_ticks); ax.grid(linestyle="--", alpha=.4)
    ax.legend(ncol=7, frameon=False, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    path = FIGURE_DIR / "问题四_典型星期电价曲线.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    lags = np.arange(1, 29)
    correlations = [metrics["ACF整日滞后"][str(lag)] for lag in lags]
    colors = ["#D95F02" if lag % 7 == 0 else "#9ECAE1" for lag in lags]
    fig, ax = plt.subplots(figsize=(9.2, 4.0), constrained_layout=True)
    ax.bar(lags, correlations, color=colors, width=.8)
    confidence = 1.96 / np.sqrt(prices.actual.size)
    ax.axhline(confidence, color="#666666", linestyle="--", linewidth=.9)
    ax.axhline(-confidence, color="#666666", linestyle="--", linewidth=.9)
    ax.set_xticks(np.arange(1, 29)); ax.set_xlabel("滞后天数"); ax.set_ylabel(r"自相关系数 $\mathrm{ACF}(144h)$")
    ax.grid(axis="y", linestyle="--", alpha=.4); ax.spines[["top", "right"]].set_visible(False)
    ax.text(.99, .96, "橙色为整周滞后，虚线为95%白噪声界限", transform=ax.transAxes,
            ha="right", va="top", color="#D95F02")
    path = FIGURE_DIR / "问题四_ACF自相关.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    names = list(metrics["0点模型比较"])
    maes = [metrics["0点模型比较"][name]["MAE(元/kWh)"] for name in names]
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    bars = ax.barh(names, maes, color=["#9ECAE1", "#6BAED6", "#4292C6", "#084594"])
    ax.invert_yaxis(); ax.set_xlabel("MAE/(元/kWh)"); ax.grid(axis="x", linestyle="--", alpha=.5)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in maes], padding=3)
    path = FIGURE_DIR / "问题四_电价预测模型误差对比.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)

    selected = detail43[detail43["日期"] == pd.Timestamp("2025-03-20")].sort_values("时段编号")
    # 与第三问指定日滚动调度图保持相同的 0--24 h、每 3 h 一主刻度形式。
    x = (np.arange(N) + 1) / 6.0
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(x, selected["实际电价(元/kWh)"], label="实际电价", color="#222222")
    axes[0].plot(x, selected["预测电价(元/kWh)"], label="滚动预测", color="#D95F02", linestyle="--")
    axes[0].set_ylabel("电价/(元/kWh)"); axes[0].legend(frameon=False)
    axes[1].plot(x, selected["计划购电量(kWh)"], label="0:00计划", color="#999999", linestyle="--")
    axes[1].plot(x, selected["调整后购电量(kWh)"], label="调整后购电", color="#1F78B4")
    axes[1].set_ylabel("购电量/kWh"); axes[1].legend(frameon=False)
    axes[2].plot(x, selected["时段末储电量(kWh)"], color="#6A3D9A")
    axes[2].set_ylabel("储电量/kWh"); axes[2].set_xlabel("时刻/h")
    for ax in axes:
        ax.set_xlim(0, 24); ax.set_xticks(hour_ticks)
        ax.grid(linestyle="--", alpha=.45); ax.spines[["top", "right"]].set_visible(False)
    path = FIGURE_DIR / "问题四_指定日价格预测与滚动调度.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="问题四周期电价与联合场景MILP")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--policies", choices=["official", "all"], default="all")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--oracle-only", action="store_true",
                        help="复用已有现实策略结果，只补算价格完全信息下界")
    parser.add_argument("--policy-comparison-only", action="store_true",
                        help="复用正式策略结果，只补算4-3各更新时点对照策略")
    parser.add_argument("--plots-only", action="store_true",
                        help="复用已有正式策略明细，只重绘论文图")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    end_date = pd.Timestamp(args.end_date)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    base = load_problem2_data(); p3 = load_problem3_data(); prices = build_price_forecasts(base.dates)
    load_forecast, pv_forecast, _ = causal_forecasts(base); p3_load, _ = load_forecast_only(p3)
    metrics = forecast_metrics(prices)

    if args.plots_only:
        detail_path = INTERMEDIATE_DIR / f"问题4-3_{OFFICIAL_POLICY}_逐时段.csv"
        summary_path = INTERMEDIATE_DIR / "问题四_汇总与校验.json"
        if not detail_path.exists() or not summary_path.exists():
            raise FileNotFoundError("--plots-only 需要先生成正式策略明细")
        detail = pd.read_csv(detail_path, encoding="utf-8-sig", parse_dates=["日期"])
        with summary_path.open("r", encoding="utf-8") as handle:
            summaries = json.load(handle)["策略汇总"]
        paths = draw_figures(prices, detail, summaries, metrics)
        write_json(INTERMEDIATE_DIR / "问题四_论文图清单.json", [str(path) for path in paths])
        print(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2), flush=True)
        return

    if args.oracle_only:
        bundle_path = INTERMEDIATE_DIR / "问题四_工作簿数据.json"
        summary_path = INTERMEDIATE_DIR / "问题四_汇总与校验.json"
        if not bundle_path.exists() or not summary_path.exists():
            raise FileNotFoundError("--oracle-only 需要先生成现实策略结果")
        with bundle_path.open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        with summary_path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        summaries = audit["策略汇总"]
        validations = audit["约束校验"]
        _, oracle42, oracle42_checks = simulate_42(
            base, prices, load_forecast, pv_forecast, end_date, oracle=True)
        _, oracle43, oracle43_checks = simulate_43(
            p3, prices, p3_load, OFFICIAL_POLICY, POLICIES[OFFICIAL_POLICY], end_date, oracle=True)
        summaries["问题4-2_完全信息价格下界"] = summarize(oracle42)
        summaries["问题4-3_完全信息价格下界"] = summarize(oracle43)
        validations["问题4-2_完全信息价格下界"] = oracle42_checks
        validations["问题4-3_完全信息价格下界"] = oracle43_checks
        for prefix in ["问题4-2", "问题4-3"]:
            reality_key = f"{prefix}_现实策略" if prefix == "问题4-2" else f"{prefix}_{OFFICIAL_POLICY}"
            reality = summaries[reality_key]["实际总费用(元)"]
            oracle = summaries[f"{prefix}_完全信息价格下界"]["实际总费用(元)"]
            summaries[f"{prefix}_价格信息价值"] = {
                "费用差(元)": reality - oracle, "相对差距": (reality - oracle) / oracle,
            }
        bundle["summaries"] = summaries
        bundle["validations"] = validations
        audit["策略汇总"] = summaries
        audit["约束校验"] = validations
        write_json(bundle_path, bundle)
        write_json(summary_path, audit)
        print(json.dumps({"策略汇总": summaries, "约束校验": validations},
                         ensure_ascii=False, indent=2), flush=True)
        return

    if args.policy_comparison_only:
        bundle_path = INTERMEDIATE_DIR / "问题四_工作簿数据.json"
        summary_path = INTERMEDIATE_DIR / "问题四_汇总与校验.json"
        if not bundle_path.exists() or not summary_path.exists():
            raise FileNotFoundError("--policy-comparison-only 需要先生成正式策略结果")
        with bundle_path.open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        with summary_path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        summaries = audit["策略汇总"]
        validations = audit["约束校验"]
        for name, updates in POLICIES.items():
            if name == OFFICIAL_POLICY:
                continue
            detail, daily, checks = simulate_43(
                p3, prices, p3_load, name, updates, end_date, oracle=False)
            detail.to_csv(INTERMEDIATE_DIR / f"问题4-3_{name}_逐时段.csv",
                          index=False, encoding="utf-8-sig")
            daily.to_csv(INTERMEDIATE_DIR / f"问题4-3_{name}_逐日.csv",
                         index=False, encoding="utf-8-sig")
            summaries[f"问题4-3_{name}"] = summarize(daily)
            validations[f"问题4-3_{name}"] = checks
        bundle["summaries"] = summaries
        bundle["validations"] = validations
        audit["策略汇总"] = summaries
        audit["约束校验"] = validations
        write_json(bundle_path, bundle)
        write_json(summary_path, audit)
        print(json.dumps({"策略汇总": summaries, "约束校验": validations},
                         ensure_ascii=False, indent=2), flush=True)
        return

    detail42, daily42, checks42 = simulate_42(base, prices, load_forecast, pv_forecast, end_date, oracle=False)
    detail42.to_csv(INTERMEDIATE_DIR / "问题4-2_现实策略逐时段.csv", index=False, encoding="utf-8-sig")
    daily42.to_csv(INTERMEDIATE_DIR / "问题4-2_现实策略逐日.csv", index=False, encoding="utf-8-sig")
    summaries = {"问题4-2_现实策略": summarize(daily42)}
    validations = {"问题4-2_现实策略": checks42}

    policies = POLICIES if args.policies == "all" else {OFFICIAL_POLICY: POLICIES[OFFICIAL_POLICY]}
    daily43_all = []
    official_detail43 = official_daily43 = None
    for name, updates in policies.items():
        detail, daily, checks = simulate_43(p3, prices, p3_load, name, updates, end_date, oracle=False)
        detail.to_csv(INTERMEDIATE_DIR / f"问题4-3_{name}_逐时段.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(INTERMEDIATE_DIR / f"问题4-3_{name}_逐日.csv", index=False, encoding="utf-8-sig")
        summaries[f"问题4-3_{name}"] = summarize(daily); validations[f"问题4-3_{name}"] = checks
        daily43_all.append(daily)
        if name == OFFICIAL_POLICY:
            official_detail43, official_daily43 = detail, daily

    if not args.skip_oracle:
        _, oracle42, oracle42_checks = simulate_42(base, prices, load_forecast, pv_forecast, end_date, oracle=True)
        summaries["问题4-2_完全信息价格下界"] = summarize(oracle42)
        validations["问题4-2_完全信息价格下界"] = oracle42_checks
        _, oracle43, oracle43_checks = simulate_43(
            p3, prices, p3_load, OFFICIAL_POLICY, POLICIES[OFFICIAL_POLICY], end_date, oracle=True)
        summaries["问题4-3_完全信息价格下界"] = summarize(oracle43)
        validations["问题4-3_完全信息价格下界"] = oracle43_checks
        for prefix in ["问题4-2", "问题4-3"]:
            reality = summaries[f"{prefix}_现实策略" if prefix == "问题4-2" else f"{prefix}_{OFFICIAL_POLICY}"]["实际总费用(元)"]
            oracle = summaries[f"{prefix}_完全信息价格下界"]["实际总费用(元)"]
            summaries[f"{prefix}_价格信息价值"] = {
                "费用差(元)": reality - oracle, "相对差距": (reality - oracle) / oracle,
            }

    if official_detail43 is None or official_daily43 is None:
        raise RuntimeError("未生成问题4-3正式策略")
    bundle = build_bundle(detail42, daily42, official_detail43, official_daily43,
                          summaries, validations, metrics)
    write_json(INTERMEDIATE_DIR / "问题四_工作簿数据.json", bundle)
    write_json(INTERMEDIATE_DIR / "问题四_汇总与校验.json", {
        "运行环境": {"Python": sys.version, "platform": platform.platform(), "NumPy": np.__version__,
                     "pandas": pd.__version__, "SciPy": scipy.__version__},
        "场景抽样次数": SCENARIO_DRAWS,
        "预测指标": metrics, "策略汇总": summaries, "约束校验": validations,
    })
    pd.DataFrame(prices.weights, columns=["滞后1周", "滞后2周", "滞后3周", "滞后4周"]).assign(
        日期=prices.dates).to_csv(INTERMEDIATE_DIR / "问题四_固定指数权重审计.csv", index=False, encoding="utf-8-sig")
    if not args.skip_plots and end_date >= OFFICIAL_END:
        paths = draw_figures(prices, official_detail43, summaries, metrics)
        write_json(INTERMEDIATE_DIR / "问题四_论文图清单.json", [str(path) for path in paths])
    print(json.dumps({"预测指标": metrics, "策略汇总": summaries, "约束校验": validations},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
