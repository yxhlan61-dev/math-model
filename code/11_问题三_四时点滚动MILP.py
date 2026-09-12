"""运行问题三四种更新策略、输出中间结果并生成论文图。"""

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

from problem2_solver_common import CHECK_TOL, ETA_C, ETA_D, N, RESERVE_TARGET, SOC_INITIAL
from problem3_solver_common import (
    FIGURE_DIR,
    INTERMEDIATE_DIR,
    ISSUE_HOURS,
    OFFICIAL_END,
    OFFICIAL_START,
    OUTPUT_DIR,
    RESERVE_PENALTY,
    SCENARIO_COUNT,
    generate_horizon_scenarios,
    load_forecast_only,
    load_problem3_data,
    solve_horizon_milp,
    validate_horizon,
    write_json,
)


POLICIES = {
    "S0_仅0点": (),
    "S6_增加6点": (6,),
    "S612_增加6点12点": (6, 12),
    "S61218_全部四时点": (6, 12, 18),
}
OFFICIAL_POLICY = "S61218_全部四时点"
SELECTED_DATE = pd.Timestamp("2025-03-20")
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
    median = float(np.median(hist)) if len(hist) >= SOC_RISK_MIN_HISTORY else float("nan")
    scale = float(max(1.4826 * np.median(np.abs(hist - median)), 100.0)) if len(hist) >= SOC_RISK_MIN_HISTORY else float("nan")
    delta = 0.0 if len(hist) < SOC_RISK_MIN_HISTORY else float(np.clip(
        SOC_RISK_GAIN * (risk_score - median) / scale, -SOC_RISK_CAP, SOC_RISK_CAP
    ))
    reference = float(np.clip(seasonal + delta, 5200.0, 8600.0))
    return {"seasonal": float(seasonal), "risk_median": median, "risk_scale": scale,
            "risk_delta": delta, "reference": reference,
            "lower": reference - SOC_BAND_HALF_WIDTH, "upper": reference + SOC_BAND_HALF_WIDTH}


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimSun", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax) -> None:
    ax.grid(True, color="#D0D0D0", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def simulate_policy(data, load_forecast: np.ndarray, policy_name: str, updates: tuple[int, ...],
                    end_date: pd.Timestamp = OFFICIAL_END) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = data.base
    price = base.price
    initial_soc = SOC_INITIAL
    detail_rows: list[dict] = []
    daily_rows: list[dict] = []
    maxima = {
        "max_abs_scenario_balance_residual_kwh": 0.0,
        "max_abs_soc_recursion_residual_kwh": 0.0,
        "max_abs_adjustment_residual_kwh": 0.0,
        "max_abs_actual_balance_residual_kwh": 0.0,
        "max_soc_link_residual_kwh": 0.0,
        "min_soc_kwh": float("inf"), "max_soc_kwh": -float("inf"),
        "max_charge_kwh": 0.0, "max_discharge_kwh": 0.0,
        "simultaneous_charge_discharge_count": 0,
        "simultaneous_up_down_count": 0,
        "max_mip_gap": 0.0,
    }
    solve_count = 0
    risk_history: list[float] = []
    max_day = int(np.searchsorted(base.dates.values, np.datetime64(end_date), side="right"))

    for d in range(1, max_day):
        date = base.dates[d]
        load_s, pv_s, probabilities, sampled = generate_horizon_scenarios(
            data, d, 0, 0, load_forecast, SCENARIO_COUNT
        )
        scenario_net = (load_s - pv_s).sum(axis=1)
        risk_score = float(np.quantile(scenario_net, 0.8) - probabilities @ scenario_net)
        soc_rule = soc_reference(pd.Timestamp(date), risk_score, risk_history)
        risk_history.append(risk_score)
        plan_solution = solve_horizon_milp(
            price, load_s, pv_s, probabilities, initial_soc, None,
            original_plan=None, reserve_target=soc_rule["lower"], reserve_penalty=RESERVE_PENALTY,
            terminal_upper_target=soc_rule["upper"], terminal_upper_penalty=RESERVE_PENALTY,
        )
        solve_count += 1
        check_list = [(plan_solution, load_s, pv_s, initial_soc, None)]
        original_plan = plan_solution.purchase.copy()
        executed_purchase = original_plan.copy()
        executed_charge = plan_solution.charge.copy()
        executed_discharge = plan_solution.discharge.copy()
        executed_soc = plan_solution.soc.copy()
        source_issue = np.zeros(N, dtype=int)
        scenario_source_counts = {"0": len(set(sampled))}

        for issue_hour in updates:
            start = issue_hour * 6
            rolling_initial_soc = float(executed_soc[start - 1])
            load_s, pv_s, probabilities, sampled = generate_horizon_scenarios(
                data, d, issue_hour, start, load_forecast, SCENARIO_COUNT
            )
            rolling = solve_horizon_milp(
                price[start:], load_s, pv_s, probabilities, rolling_initial_soc, None,
                original_plan=original_plan[start:], reserve_target=soc_rule["lower"],
                reserve_penalty=RESERVE_PENALTY,
                terminal_upper_target=soc_rule["upper"], terminal_upper_penalty=RESERVE_PENALTY,
            )
            solve_count += 1
            check_list.append((rolling, load_s, pv_s, rolling_initial_soc, original_plan[start:]))
            executed_purchase[start:] = rolling.purchase
            executed_charge[start:] = rolling.charge
            executed_discharge[start:] = rolling.discharge
            executed_soc[start:] = rolling.soc
            source_issue[start:] = issue_hour
            scenario_source_counts[str(issue_hour)] = len(set(sampled))

        actual_emergency = np.maximum(
            base.load_kwh[d] + executed_charge - executed_purchase - base.pv_kwh[d] - executed_discharge, 0.0
        )
        actual_surplus = np.maximum(
            executed_purchase + base.pv_kwh[d] + executed_discharge - base.load_kwh[d] - executed_charge, 0.0
        )
        actual_balance = (executed_purchase + actual_emergency + base.pv_kwh[d] - actual_surplus
                          + executed_discharge - base.load_kwh[d] - executed_charge)
        upward = np.maximum(executed_purchase - original_plan, 0.0)
        downward = np.maximum(original_plan - executed_purchase, 0.0)
        plan_cost = float(price @ original_plan)
        upward_cost = float((1.5 * price) @ upward)
        downward_credit = float((-0.5 * price) @ downward)
        scheduled_cost = plan_cost + upward_cost + downward_credit
        emergency_cost = float((5.0 * price) @ actual_emergency)
        total_cost = scheduled_cost + emergency_cost
        previous_soc = np.r_[initial_soc, executed_soc[:-1]]
        executed_soc_residual = executed_soc - previous_soc - ETA_C * executed_charge + executed_discharge / ETA_D

        for solution, ls, ps, horizon_initial, horizon_plan in check_list:
            check = validate_horizon(solution, ls, ps, horizon_initial, horizon_plan)
            for key in ["max_abs_scenario_balance_residual_kwh", "max_abs_soc_recursion_residual_kwh",
                        "max_abs_adjustment_residual_kwh", "max_charge_kwh", "max_discharge_kwh"]:
                maxima[key] = max(maxima[key], float(check[key]))
            maxima["min_soc_kwh"] = min(maxima["min_soc_kwh"], float(check["min_soc_kwh"]))
            maxima["max_soc_kwh"] = max(maxima["max_soc_kwh"], float(check["max_soc_kwh"]))
            maxima["simultaneous_charge_discharge_count"] += int(check["simultaneous_charge_discharge_count"])
            maxima["simultaneous_up_down_count"] += int(check["simultaneous_up_down_count"])
            maxima["max_mip_gap"] = max(maxima["max_mip_gap"], float(check["mip_gap"] or 0.0))
        maxima["max_abs_actual_balance_residual_kwh"] = max(
            maxima["max_abs_actual_balance_residual_kwh"], float(np.max(np.abs(actual_balance)))
        )
        maxima["max_abs_soc_recursion_residual_kwh"] = max(
            maxima["max_abs_soc_recursion_residual_kwh"], float(np.max(np.abs(executed_soc_residual)))
        )
        if daily_rows:
            maxima["max_soc_link_residual_kwh"] = max(
                maxima["max_soc_link_residual_kwh"], abs(initial_soc - daily_rows[-1]["日末储电量(kWh)"])
            )

        daily_rows.append({
            "策略": policy_name, "日期": date, "日初储电量(kWh)": initial_soc,
            "日末储电量(kWh)": float(executed_soc[-1]), "计划购电量(kWh)": float(original_plan.sum()),
            "调整后购电量(kWh)": float(executed_purchase.sum()), "上调量(kWh)": float(upward.sum()),
            "下调量(kWh)": float(downward.sum()), "紧急购电量(kWh)": float(actual_emergency.sum()),
            "富余电量(kWh)": float(actual_surplus.sum()), "充电量(kWh)": float(executed_charge.sum()),
            "放电量(kWh)": float(executed_discharge.sum()), "计划购电费(元)": plan_cost,
            "上调费用(元)": upward_cost, "下调抵扣(元)": downward_credit,
            "计划与调整结算费(元)": scheduled_cost, "紧急购电费(元)": emergency_cost,
            "实际总费用(元)": total_cost,
            "SOC季节基准(kWh)": soc_rule["seasonal"], "SOC风险分数(kWh)": risk_score,
            "SOC每日微调(kWh)": soc_rule["risk_delta"], "SOC软区间下沿(kWh)": soc_rule["lower"],
            "SOC软区间上沿(kWh)": soc_rule["upper"],
            "6点调整量(kWh)": float(np.sum(np.abs(executed_purchase[source_issue == 6] - original_plan[source_issue == 6]))),
            "12点调整量(kWh)": float(np.sum(np.abs(executed_purchase[source_issue == 12] - original_plan[source_issue == 12]))),
            "18点调整量(kWh)": float(np.sum(np.abs(executed_purchase[source_issue == 18] - original_plan[source_issue == 18]))),
            "求解次数": len(check_list), "场景历史日数": json.dumps(scenario_source_counts, ensure_ascii=False),
        })
        for t in range(N):
            detail_rows.append({
                "策略": policy_name, "日期": date, "时段编号": t + 1, "时段": base.interval_labels[t],
                "决策来源时点": int(source_issue[t]), "实际负荷(kWh)": base.load_kwh[d, t],
                "预测负荷(kWh)": load_forecast[d, t], "实际光伏(kWh)": base.pv_kwh[d, t],
                "0点预测光伏(kWh)": data.pv_interval_forecast_kwh[d, 0, t],
                "最终来源预测光伏(kWh)": data.pv_interval_forecast_kwh[d, ISSUE_HOURS.index(int(source_issue[t])), t],
                "计划购电量(kWh)": original_plan[t], "调整后购电量(kWh)": executed_purchase[t],
                "上调量(kWh)": upward[t], "下调量(kWh)": downward[t],
                "充电量(kWh)": executed_charge[t], "放电量(kWh)": executed_discharge[t],
                "时段末储电量(kWh)": executed_soc[t], "实际紧急购电量(kWh)": actual_emergency[t],
                "实际富余电量(kWh)": actual_surplus[t], "电价(元/kWh)": price[t],
            })
        initial_soc = float(executed_soc[-1])
        if d % 10 == 0 or d == max_day - 1:
            print(f"[{policy_name}] {date.date()} 完成，累计求解 {solve_count} 次", flush=True)

    detail = pd.DataFrame(detail_rows)
    daily = pd.DataFrame(daily_rows)
    return detail, daily, maxima


def summarize_policy(daily: pd.DataFrame) -> dict:
    official = daily[(daily["日期"] >= OFFICIAL_START) & (daily["日期"] <= OFFICIAL_END)]
    sum_columns = ["计划购电量(kWh)", "调整后购电量(kWh)", "上调量(kWh)", "下调量(kWh)",
                   "紧急购电量(kWh)", "富余电量(kWh)", "充电量(kWh)", "放电量(kWh)",
                   "计划购电费(元)", "上调费用(元)", "下调抵扣(元)", "计划与调整结算费(元)",
                   "紧急购电费(元)", "实际总费用(元)"]
    result = {column: float(official[column].sum()) for column in sum_columns}
    result.update({
        "正式天数": int(len(official)), "平均日末储电量(kWh)": float(official["日末储电量(kWh)"].mean()),
        "日末低于SOC软区间下沿天数": int((official["日末储电量(kWh)"] < official["SOC软区间下沿(kWh)"] - CHECK_TOL).sum()),
        "发生紧急购电天数": int((official["紧急购电量(kWh)"] > CHECK_TOL).sum()),
        "6点发生调整天数": int((official["6点调整量(kWh)"] > CHECK_TOL).sum()),
        "12点发生调整天数": int((official["12点调整量(kWh)"] > CHECK_TOL).sum()),
        "18点发生调整天数": int((official["18点调整量(kWh)"] > CHECK_TOL).sum()),
        "12月31日末储电量(kWh)": float(official.iloc[-1]["日末储电量(kWh)"]),
    })
    return result


def forecast_metrics(data) -> dict:
    remaining = {}
    for issue_index, issue_hour in enumerate(ISSUE_HOURS):
        start = issue_hour * 6
        actual = data.base.pv_kwh[:, start:]
        forecast = data.pv_interval_forecast_kwh[:, issue_index, start:]
        mask = np.isfinite(forecast)
        error = actual[mask] - forecast[mask]
        nonzero = mask & (actual > CHECK_TOL)
        error_nz = actual[nonzero] - forecast[nonzero]
        remaining[str(issue_hour)] = {
            "MAE(kWh)": float(np.mean(np.abs(error))), "RMSE(kWh)": float(np.sqrt(np.mean(error ** 2))),
            "非零时段MAE(kWh)": float(np.mean(np.abs(error_nz))), "样本数": int(mask.sum()),
        }
    same_target = {}
    for target_hour in (6, 12, 18):
        start, end = target_hour * 6, min((target_hour + 6) * 6, N)
        block = {}
        for issue_hour in [hour for hour in ISSUE_HOURS if hour <= target_hour]:
            issue_index = ISSUE_HOURS.index(issue_hour)
            actual = data.base.pv_kwh[:, start:end]
            forecast = data.pv_interval_forecast_kwh[:, issue_index, start:end]
            error = actual - forecast
            block[str(issue_hour)] = {
                "MAE(kWh)": float(np.mean(np.abs(error))),
                "RMSE(kWh)": float(np.sqrt(np.mean(error ** 2))),
                "样本数": int(error.size),
            }
        same_target[f"{target_hour}-{target_hour + 6}"] = block
    return {"各发布时间剩余时域": remaining, "同目标6小时区段": same_target}


def build_bundle(data, official_detail: pd.DataFrame, official_daily: pd.DataFrame,
                 policy_summaries: dict, validations: dict, forecasts: dict) -> dict:
    detail = official_detail[(official_detail["日期"] >= OFFICIAL_START) & (official_detail["日期"] <= OFFICIAL_END)]
    daily = official_daily[(official_daily["日期"] >= OFFICIAL_START) & (official_daily["日期"] <= OFFICIAL_END)]
    purchase_rows = []
    adjustment_rows = []
    storage_rows = []
    emergency_rows = []
    for date, group in detail.groupby("日期", sort=True):
        group = group.sort_values("时段编号")
        day = daily[daily["日期"] == date].iloc[0]
        purchase_rows.append({"date": date.strftime("%Y-%m-%d"), "values": group["计划购电量(kWh)"].tolist(),
                              "total": day["计划购电量(kWh)"], "cost": day["计划购电费(元)"]})
        adjustment_rows.append({"date": date.strftime("%Y-%m-%d"), "values": group["调整后购电量(kWh)"].tolist(),
                                "total": day["调整后购电量(kWh)"], "cost": day["计划与调整结算费(元)"]})
        storage_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "charge_4h": [float(group.iloc[i:i + 24]["充电量(kWh)"].sum()) for i in range(0, N, 24)],
            "discharge_4h": [float(group.iloc[i:i + 24]["放电量(kWh)"].sum()) for i in range(0, N, 24)],
            "initial_soc": float(day["日初储电量(kWh)"]), "terminal_soc": float(day["日末储电量(kWh)"]),
        })
        values = group["实际紧急购电量(kWh)"].to_numpy()
        active = values > CHECK_TOL
        start = None
        labels = group["时段"].tolist()
        for t in range(N + 1):
            on = bool(active[t]) if t < N else False
            if on and start is None:
                start = t
            elif not on and start is not None:
                emergency_rows.append({"date": date.strftime("%Y-%m-%d"),
                                       "period": f"{labels[start].split('-')[0]}-{labels[t-1].split('-')[1]}",
                                       "amount": float(values[start:t].sum())})
                start = None
    return {
        "purchaseRows": purchase_rows, "adjustmentRows": adjustment_rows, "storageRows": storage_rows,
        "emergencyRows": emergency_rows, "policySummaries": policy_summaries, "validations": validations,
        "forecastMetrics": forecasts,
        "dailyRows": [{k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v) for k, v in row.items()}
                      for row in daily.to_dict("records")],
    }


def draw_figures(data, all_daily: pd.DataFrame, official_detail: pd.DataFrame, forecasts: dict) -> list[Path]:
    configure_plotting(); FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    official_daily = all_daily[(all_daily["日期"] >= OFFICIAL_START) & (all_daily["日期"] <= OFFICIAL_END)]
    order = list(POLICIES)
    colors = ["#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
    paths = []

    totals = official_daily.groupby("策略")[["计划与调整结算费(元)", "紧急购电费(元)"]].sum().reindex(order)
    official_period_detail = official_detail[
        (official_detail["日期"] >= OFFICIAL_START) & (official_detail["日期"] <= OFFICIAL_END)
    ]
    period_price = official_period_detail["电价(元/kWh)"].to_numpy(dtype=float)
    adjusted_purchase = official_period_detail["调整后购电量(kWh)"].to_numpy(dtype=float)
    upward_adjustment = official_period_detail["上调量(kWh)"].to_numpy(dtype=float)
    downward_adjustment = official_period_detail["下调量(kWh)"].to_numpy(dtype=float)
    actual_emergency = official_period_detail["实际紧急购电量(kWh)"].to_numpy(dtype=float)
    cost_values = np.array([
        np.sum(period_price * adjusted_purchase),
        np.sum(0.5 * period_price * upward_adjustment),
        np.sum(0.5 * period_price * downward_adjustment),
        np.sum(5.0 * period_price * actual_emergency),
    ])
    cost_labels = ["调整后购电基本费", "上调超出加价费", "下调违约费", "紧急购电费"]
    total_final_cost = float(cost_values.sum())
    cost_shares = cost_values / total_final_cost * 100
    legend_labels = [
        f"{label}：{value / 10_000:.2f}万元（{share:.2f}%）"
        for label, value, share in zip(cost_labels, cost_values, cost_shares)
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    wedges, _, autotexts = ax.pie(
        cost_values,
        colors=["#3182BD", "#EDC948", "#E15759", "#F28E2B"],
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.2f}%" if pct >= 3 else "",
        pctdistance=0.76,
        explode=[0.0, 0.12, 0.08, 0.04],
        wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 11},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
    small_slice_annotations = {
        1: ("上调超出", (-1.20, 0.84)),
        2: ("下调违约", (-1.20, 1.08)),
    }
    for index, (short_label, text_position) in small_slice_annotations.items():
        wedge = wedges[index]
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(angle), np.sin(angle)
        ax.annotate(
            f"{short_label}  {cost_shares[index]:.2f}%",
            xy=(0.98 * x, 0.98 * y),
            xytext=text_position,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 0.9},
        )
    ax.text(0, 0.05, "全年实际总费用", ha="center", va="center", fontsize=11)
    ax.text(0, -0.10, f"{total_final_cost / 10_000:.2f}万元", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#333333")
    ax.legend(wedges, legend_labels, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9.5,
              columnspacing=1.6, handlelength=1.5)
    ax.set_aspect("equal")
    p = FIGURE_DIR / "问题三_最终四时点策略费用构成占比.pdf"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    total_cost = totals.sum(axis=1).to_numpy()
    emergency = official_daily.groupby("策略")["紧急购电量(kWh)"].sum().reindex(order).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    marginal_cost_saving = -np.diff(total_cost)
    marginal_emergency_reduction = -np.diff(emergency)
    bars0 = axes[0].bar(["6:00", "12:00", "18:00"], marginal_cost_saving,
                        color=["#2C7FB8", "#41AB5D", "#756BB1"])
    axes[0].set_ylim(bottom=0); axes[0].set_ylabel("边际费用节省/元")
    axes[0].ticklabel_format(axis="y", style="plain")
    axes[0].bar_label(bars0, labels=[f"{value:,.0f}" for value in marginal_cost_saving],
                      padding=3, fontsize=9)
    bars1 = axes[1].bar(["6:00", "12:00", "18:00"], marginal_emergency_reduction,
                        color=["#2C7FB8", "#41AB5D", "#756BB1"])
    axes[1].set_ylim(bottom=0); axes[1].set_ylabel("边际紧急购电减少量/kWh")
    axes[1].bar_label(bars1, labels=[f"{value:,.0f}" for value in marginal_emergency_reduction],
                      padding=3, fontsize=9)
    for ax in axes: style_axis(ax); ax.set_xlabel("新增预报时点")
    p = FIGURE_DIR / "问题三_各更新时点边际价值.pdf"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.2), constrained_layout=True, sharey=True)
    same_target = forecasts["同目标6小时区段"]
    for ax, block_name in zip(axes, ["6-12", "12-18", "18-24"]):
        block = same_target[block_name]
        issue_labels = [f"{hour}:00" for hour in map(int, block.keys())]
        metric_values = [value["MAE(kWh)"] * 6 for value in block.values()]
        ax.bar(issue_labels, metric_values, color=colors[:len(metric_values)])
        ax.set_xlabel(f"目标时段 {block_name}:00"); style_axis(ax)
    axes[0].set_ylabel("光伏预测MAE/kW")
    p = FIGURE_DIR / "问题三_各发布时间光伏预测误差.pdf"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    d = int(np.where(data.base.dates == SELECTED_DATE)[0][0])
    xh = (np.arange(N) + 1) / 6
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.plot(xh, data.base.pv_kwh[d] * 6, color="#222222", linewidth=1.7, label="实际光伏")
    for issue_index, issue_hour in enumerate(ISSUE_HOURS[:3]):
        start = issue_hour * 6
        ax.plot(xh[start:], data.pv_interval_forecast_kwh[d, issue_index, start:] * 6,
                linewidth=1.25, linestyle="--", label=f"{issue_hour}:00预报")
    ax.set_xlim(0, 24); ax.set_xticks(np.arange(0, 25, 3)); ax.set_xlabel("时刻/h"); ax.set_ylabel("光伏功率/kW")
    style_axis(ax); ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    p = FIGURE_DIR / "问题三_指定日滚动光伏预报对比.pdf"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    selected = official_detail[official_detail["日期"] == SELECTED_DATE].sort_values("时段编号")
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.8), sharex=True, constrained_layout=True)
    axes[0].plot(xh, selected["计划购电量(kWh)"], color="#8C8C8C", linestyle="--", label="0:00计划购电")
    axes[0].plot(xh, selected["调整后购电量(kWh)"], color="#08519C", label="最终调整购电")
    axes[0].plot(xh, selected["实际紧急购电量(kWh)"], color="#D62728", label="紧急购电")
    axes[0].set_ylabel("购电量/kWh"); axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(xh, selected["充电量(kWh)"], color="#2CA02C", label="充电量")
    axes[1].plot(xh, -selected["放电量(kWh)"], color="#FF7F0E", label="放电量（负向）")
    axes[1].set_ylabel("充放电量/kWh"); axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(xh, selected["时段末储电量(kWh)"], color="#6A3D9A", linewidth=1.5)
    axes[2].set_ylabel("储电量/kWh"); axes[2].set_xlabel("时刻/h")
    for ax in axes: style_axis(ax); ax.set_xlim(0, 24); ax.set_xticks(np.arange(0, 25, 3))
    p = FIGURE_DIR / "问题三_指定日滚动调度结果.pdf"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="问题三四时点滚动MILP")
    parser.add_argument("--policy", choices=["all", *POLICIES], default="all")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    end_date = pd.Timestamp(args.end_date)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_problem3_data(); load_forecast, forecast_audit = load_forecast_only(data)
    policies = POLICIES if args.policy == "all" else {args.policy: POLICIES[args.policy]}
    details = {}; daily_frames = []; validations = {}; summaries = {}
    for name, updates in policies.items():
        detail, daily, checks = simulate_policy(data, load_forecast, name, updates, end_date=end_date)
        details[name] = detail; daily_frames.append(daily); validations[name] = checks
        summaries[name] = summarize_policy(daily) if end_date >= OFFICIAL_END else {}
        detail.to_csv(INTERMEDIATE_DIR / f"{name}_逐时段结果.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(INTERMEDIATE_DIR / f"{name}_逐日汇总.csv", index=False, encoding="utf-8-sig")
        write_json(INTERMEDIATE_DIR / f"{name}_约束校验.json", checks)
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily.to_csv(INTERMEDIATE_DIR / "问题三_四策略逐日汇总.csv", index=False, encoding="utf-8-sig")
    forecasts = forecast_metrics(data)
    metadata = {
        "运行环境": {"Python": sys.version, "platform": platform.platform(), "NumPy": np.__version__,
                     "pandas": pd.__version__, "SciPy": scipy.__version__},
        "参数": {"场景数": SCENARIO_COUNT, "日备目标_kWh": RESERVE_TARGET,
                 "日备不足惩罚_元每kWh": RESERVE_PENALTY, "更新策略": {k: list(v) for k, v in POLICIES.items()}},
        "光伏预测指标": forecasts, "策略汇总": summaries, "约束校验": validations,
    }
    write_json(INTERMEDIATE_DIR / "问题三_汇总与校验.json", metadata)
    pd.DataFrame(forecast_audit).to_csv(INTERMEDIATE_DIR / "问题三_负荷预测信息边界审计.csv", index=False, encoding="utf-8-sig")

    if OFFICIAL_POLICY in details and end_date >= OFFICIAL_END:
        official_detail = details[OFFICIAL_POLICY]
        official_daily = all_daily[all_daily["策略"] == OFFICIAL_POLICY]
        bundle = build_bundle(data, official_detail, official_daily, summaries, validations, forecasts)
        write_json(INTERMEDIATE_DIR / "问题三_工作簿数据.json", bundle)
        if not args.skip_plots and len(policies) == 4:
            paths = draw_figures(data, all_daily, official_detail, forecasts)
            metadata["论文图"] = [str(path) for path in paths]
            write_json(INTERMEDIATE_DIR / "问题三_汇总与校验.json", metadata)
    print(json.dumps({"策略汇总": summaries, "约束校验": validations}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
