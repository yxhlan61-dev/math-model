"""运行问题二全年因果预测、联合场景 MILP、实际结算并生成论文图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.stattools import acf

from problem2_solver_common import (
    CHECK_TOL,
    ENERGY_MAX,
    ETA_D,
    FIGURE_DIR,
    INTERMEDIATE_DIR,
    N,
    OFFICIAL_END,
    OFFICIAL_START,
    OUTPUT_DIR,
    RHO,
    RESERVE_TARGET,
    ROOT,
    SCENARIO_COUNT,
    SOC_INITIAL,
    causal_forecasts,
    generate_scenarios,
    load_problem2_data,
    merge_emergency_intervals,
    prediction_metrics,
    solve_daily_milp,
    validate_daily_solution,
    write_json,
)


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        # 固定使用完整覆盖常用中文字符的字体，避免 PDF 阅读器对个别汉字发生异色回退。
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


def _style_axis(ax) -> None:
    ax.grid(True, color="#D0D0D0", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_yearly_figure(detail: pd.DataFrame) -> Path:
    """只重绘全年购电与日末储电量图。"""
    _configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # 全年展示包含 1 月预热求解结果；逐时段结果从 1 月 2 日开始，横轴仍从 1 月 1 日起。
    yearly = detail[(detail["日期"] >= pd.Timestamp("2025-01-01")) & (detail["日期"] <= OFFICIAL_END)]
    daily = yearly.groupby("日期", as_index=False).agg(
        正常购电量=("计划购电量(kWh)", "sum"),
        紧急购电量=("实际紧急购电量(kWh)", "sum"),
        日末储电量=("时段末储电量(kWh)", "last"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), sharex=True, constrained_layout=True)
    purchase_ax = axes[0]
    purchase_ax.plot(daily["日期"], daily["正常购电量"], color="#1F77B4", linewidth=1.0, label="正常购电量")
    purchase_ax.fill_between(daily["日期"], 0, daily["紧急购电量"], color="#D62728", alpha=0.40,
                             linewidth=0, label="紧急购电量")
    purchase_ax.set_ylabel("购电量/(kWh/日)")
    purchase_ax.set_ylim(bottom=0)
    _style_axis(purchase_ax)
    purchase_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015),
                       ncol=2, frameon=False, borderaxespad=0.0)
    axes[1].plot(daily["日期"], daily["日末储电量"], color="#08519C", linewidth=1.25,
                 marker="o", markevery=14, markersize=2.5, label="日末储电量")
    axes[1].set_ylabel("日末储电量/kWh")
    axes[1].set_xlabel("日期")
    date_ticks = pd.date_range("2025-01-01", "2025-12-01", freq="MS").append(
        pd.DatetimeIndex([pd.Timestamp("2025-12-31")])
    )
    axes[1].set_xticks(date_ticks)
    axes[1].xaxis.set_major_formatter(FuncFormatter(
        lambda value, _position: f"{mdates.num2date(value).month}.{mdates.num2date(value).day}"
    ))
    axes[1].set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    _style_axis(axes[1])
    axes[1].set_ylim(0, 12000)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=1, frameon=False)
    path = FIGURE_DIR / "问题二_全年正常购电紧急购电与储电量.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_weekday_figure(data) -> Path:
    """严格按 2025 年公历星期绘制平均负荷与光伏曲线。"""
    _configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hours = (np.arange(N) + 1.5) / 6.0
    hour_ticks = np.arange(0, 25, 3)
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    # 星期五、星期六是实际公历分组后的两条低负荷曲线，用暖色突出；
    # 其余五天统一使用蓝绿色系，避免把星期日误读为低负荷日。
    colors = ["#2166AC", "#4393C3", "#67A9CF", "#1B9E77", "#F28E2B", "#D62728", "#00A087"]
    calendar_weekday = data.dates.weekday

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), sharex=True, constrained_layout=True)
    for weekday, name in enumerate(weekday_names):
        mask = calendar_weekday == weekday
        is_low_load_day = weekday in (4, 5)
        axes[0].plot(hours, data.load_kwh[mask].mean(axis=0) * 6, label=name, color=colors[weekday],
                     linewidth=1.8 if is_low_load_day else 1.15,
                     linestyle="--" if is_low_load_day else "-")
        axes[1].plot(hours, data.pv_kwh[mask].mean(axis=0) * 6, label=name, color=colors[weekday],
                     linewidth=1.8 if is_low_load_day else 1.15,
                     linestyle="--" if is_low_load_day else "-")
    axes[0].set_ylabel("平均负荷功率/kW")
    axes[1].set_ylabel("平均光伏功率/kW")
    axes[1].set_xlabel("时刻/h")
    axes[1].set_xticks(hour_ticks)
    for ax in axes:
        ax.set_xlim(0, 24)
        _style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.015),
               frameon=False, columnspacing=0.9, handlelength=1.8)
    path = FIGURE_DIR / "问题二_星期一至星期日平均负荷与光伏曲线.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_comparison(current_daily: pd.DataFrame, baseline_daily: pd.DataFrame) -> dict:
    """汇总有、无日备不足惩罚两种同口径方案。"""
    fields = {
        "计划购电量_kWh": "计划购电量(kWh)",
        "实际紧急购电量_kWh": "实际紧急购电量(kWh)",
        "充电量_kWh": "充电量(kWh)",
        "放电量_kWh": "放电量(kWh)",
        "正常购电费_元": "正常购电费(元)",
        "紧急购电费_元": "紧急购电费(元)",
        "实际购电总费用_元": "实际总费用(元)",
    }

    def summarize(frame: pd.DataFrame) -> dict:
        official = frame[(frame["日期"] >= OFFICIAL_START) & (frame["日期"] <= OFFICIAL_END)]
        result = {name: float(official[column].sum()) for name, column in fields.items()}
        result.update({
            "平均日末储电量_kWh": float(official["日末储电量(kWh)"].mean()),
            "日末储电量标准差_kWh": float(official["日末储电量(kWh)"].std(ddof=0)),
            "日末低于6000kWh天数": int((official["日末储电量(kWh)"] < RESERVE_TARGET - CHECK_TOL).sum()),
            "12月31日末储电量_kWh": float(official.iloc[-1]["日末储电量(kWh)"]),
        })
        return result

    without = summarize(baseline_daily)
    with_reserve = summarize(current_daily)
    delta = {key: with_reserve[key] - without[key] for key in without}
    return {
        "对比口径": "两方案均取消CVaR与富余电量极小惩罚，仅日备不足惩罚不同",
        "无日备不足惩罚": without,
        "有日备不足惩罚": with_reserve,
        "差值_有减无": delta,
    }


def draw_comparison_figure(current_daily: pd.DataFrame, baseline_daily: pd.DataFrame) -> Path:
    """费用只绘制差额，日末储电量保留两套方案的绝对值。"""
    _configure_matplotlib()
    current_daily = current_daily.copy()
    baseline_daily = baseline_daily.copy()
    current_daily["月份"] = current_daily["日期"].dt.to_period("M").astype(str)
    baseline_daily["月份"] = baseline_daily["日期"].dt.to_period("M").astype(str)
    current_monthly = current_daily[current_daily["日期"] >= OFFICIAL_START].groupby("月份")["实际总费用(元)"].sum()
    baseline_monthly = baseline_daily[baseline_daily["日期"] >= OFFICIAL_START].groupby("月份")["实际总费用(元)"].sum()
    monthly_delta = current_monthly - baseline_monthly
    months = current_monthly.index.tolist()
    x = np.arange(len(months))

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), constrained_layout=True)
    bar_colors = np.where(monthly_delta.to_numpy() <= 0, "#2C7FB8", "#D95F02")
    axes[0].bar(x, monthly_delta.values, width=0.66, color=bar_colors)
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    axes[0].set_xticks(x, [month.replace("2025-", "") + "月" for month in months])
    axes[0].set_ylabel("实际购电总费用差额/元")
    axes[0].ticklabel_format(axis="y", style="plain")
    _style_axis(axes[0])
    axes[0].text(0.01, 0.97, "差值 = 有日备不足惩罚 - 无日备不足惩罚",
                 transform=axes[0].transAxes, ha="left", va="top", color="#444444")

    axes[1].plot(
        baseline_daily["日期"], baseline_daily["日末储电量(kWh)"],
        color="#8C8C8C", linewidth=1.35, drawstyle="steps-post", label="无日备不足惩罚",
    )
    axes[1].plot(
        current_daily["日期"], current_daily["日末储电量(kWh)"],
        color="#08519C", linewidth=1.45, drawstyle="steps-post", label="有日备不足惩罚",
    )
    date_ticks = pd.date_range("2025-01-01", "2025-12-01", freq="MS").append(
        pd.DatetimeIndex([pd.Timestamp("2025-12-31")])
    )
    axes[1].set_xticks(date_ticks)
    axes[1].xaxis.set_major_formatter(FuncFormatter(
        lambda value, _position: f"{mdates.num2date(value).month}.{mdates.num2date(value).day}"
    ))
    axes[1].set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    axes[1].set_ylim(0, 6800)
    axes[1].set_ylabel("日末储电量/kWh")
    axes[1].set_xlabel("日期")
    _style_axis(axes[1])
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    path = FIGURE_DIR / "问题二_日备不足惩罚前后费用与储能对比.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_intraday_figure(detail: pd.DataFrame, selected_date: pd.Timestamp = pd.Timestamp("2025-03-20")) -> Path:
    """紧凑绘制指定日的预测功率与实际调度过程。"""
    _configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    selected = detail[detail["日期"] == selected_date].sort_values("时段编号")
    if len(selected) != N:
        raise ValueError(f"{selected_date.date()} 应有 {N} 个时段，实际读取 {len(selected)} 个")

    hours = (np.arange(N) + 1.5) / 6.0
    hour_ticks = np.arange(0, 25, 3)
    fig, axes = plt.subplots(
        2, 1, figsize=(9.2, 6.8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.05, 0.95]},
    )

    # 预测层：负荷与光伏都是功率，可安全合并到同一坐标轴。
    axes[0].plot(hours, selected["实际负荷(kWh)"] * 6, color="#222222",
                 linewidth=1.45, label="实际负荷")
    axes[0].plot(hours, selected["预测负荷(kWh)"] * 6, color="#1F77B4",
                 linestyle="--", linewidth=1.35, label="预测负荷")
    axes[0].plot(hours, selected["实际光伏(kWh)"] * 6, color="#1B9E77",
                 linewidth=1.45, label="实际光伏")
    axes[0].plot(hours, selected["预测光伏(kWh)"] * 6, color="#E69F00",
                 linestyle="--", linewidth=1.35, label="预测光伏")

    # 实际过程层：储电量、正常购电量与紧急购电量均以 kWh 计量。
    axes[1].plot(hours, selected["时段末储电量(kWh)"], color="#08519C",
                 linewidth=1.55, label="储电量")
    axes[1].plot(
        hours, selected["计划购电量(kWh)"], color="#2CA25F", linewidth=1.35,
        drawstyle="steps-mid", label="正常购电量",
    )
    axes[1].plot(
        hours, selected["实际紧急购电量(kWh)"], color="#D62728", linewidth=1.6,
        drawstyle="steps-mid", label="紧急购电量",
    )
    axes[1].fill_between(
        hours, 0, selected["实际紧急购电量(kWh)"].to_numpy(), step="mid",
        color="#D62728", alpha=0.16,
    )
    axes[0].set_ylabel("功率/kW")
    axes[1].set_ylabel("电量/kWh")
    axes[1].set_xlabel("时刻/h")
    axes[1].set_xticks(hour_ticks)
    for ax in axes:
        ax.set_xlim(0, 24)
        _style_axis(ax)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=4,
                   frameon=False, columnspacing=1.1, handlelength=2.1)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=3,
                   frameon=False, columnspacing=1.4, handlelength=2.2)
    axes[0].text(0.01, 0.96, selected_date.strftime("%Y年%m月%d日"),
                 transform=axes[0].transAxes, ha="left", va="top", color="#444444")
    path = FIGURE_DIR / "问题二_24小时预测值与实际值.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_figures(data, detail: pd.DataFrame, load_forecast: np.ndarray, pv_forecast: np.ndarray) -> list[Path]:
    _configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.append(draw_weekday_figure(data))

    # 先把同一天的 144 个时段取平均，再计算 0~14 天的日尺度 ACF；
    # 因此图中只有覆盖两周的 15 根柱，而不是密集的 10 分钟级柱。
    max_lag_days = 14
    daily_mean_load = data.load_kwh.mean(axis=1)
    daily_mean_pv = data.pv_kwh.mean(axis=1)
    load_acf = acf(daily_mean_load, nlags=max_lag_days, fft=True)
    pv_acf = acf(daily_mean_pv, nlags=max_lag_days, fft=True)
    confidence = 1.96 / np.sqrt(len(daily_mean_load))
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True, constrained_layout=True)
    lag_days = np.arange(max_lag_days + 1)
    for ax, values, ylabel, color in [
        (axes[0], load_acf, "负荷自相关系数", "#1F77B4"),
        (axes[1], pv_acf, "光伏自相关系数", "#E69F00"),
    ]:
        ax.bar(lag_days, values, width=0.68, color=color, edgecolor="white", linewidth=0.4, alpha=0.9)
        ax.axhspan(-confidence, confidence, color="#BDBDBD", alpha=0.25)
        ax.axvline(1, color="#D62728", linestyle="--", linewidth=1.0, label="1天滞后")
        ax.axvline(7, color="#7B2CBF", linestyle="--", linewidth=1.0, label="7天滞后")
        ax.axvline(14, color="#00876C", linestyle="--", linewidth=1.0, label="14天滞后")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-1.05, 1.05)
        _style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=3, frameon=False)
    axes[1].set_xlabel("滞后时间/天")
    axes[1].set_xlim(-0.6, 14.6)
    axes[1].set_xticks(np.arange(0, 15, 1))
    path = FIGURE_DIR / "问题二_负荷与光伏ACF.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig); paths.append(path)

    paths.append(draw_intraday_figure(detail))

    march = detail[(detail["日期"] >= pd.Timestamp("2025-03-01")) & (detail["日期"] <= pd.Timestamp("2025-03-31"))]
    march_daily = march.groupby("日期", as_index=False)[["实际负荷(kWh)", "预测负荷(kWh)", "实际光伏(kWh)", "预测光伏(kWh)"]].sum()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True, constrained_layout=True)
    axes[0].plot(march_daily["日期"], march_daily["实际负荷(kWh)"], color="#222222", marker="o", markersize=2.8, label="实际值")
    axes[0].plot(march_daily["日期"], march_daily["预测负荷(kWh)"], color="#1F77B4", linestyle="--", marker="o", markersize=2.4, label="预测值")
    axes[1].plot(march_daily["日期"], march_daily["实际光伏(kWh)"], color="#222222", marker="o", markersize=2.8, label="实际值")
    axes[1].plot(march_daily["日期"], march_daily["预测光伏(kWh)"], color="#E69F00", linestyle="--", marker="o", markersize=2.4, label="预测值")
    axes[0].set_ylabel("日负荷用电量/kWh")
    axes[1].set_ylabel("日光伏发电量/kWh")
    axes[1].set_xlabel("日期")
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for ax in axes:
        _style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False)
    path = FIGURE_DIR / "问题二_一个月预测值与实际值.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig); paths.append(path)

    paths.append(draw_yearly_figure(detail))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二日备不足惩罚MILP全年求解")
    parser.add_argument(
        "--variant", choices=["reserve", "no-reserve"], default="reserve",
        help="reserve为最终日备惩罚方案，no-reserve为同口径无惩罚对照方案",
    )
    parser.add_argument(
        "--reserve-penalty", type=float, default=None,
        help="日备不足惩罚系数（元/kWh）；默认取放电效率乘固定电价90%%分位数",
    )
    args = parser.parse_args()
    data = load_problem2_data()
    reserve_penalty = (
        0.0 if args.variant == "no-reserve"
        else float(args.reserve_penalty if args.reserve_penalty is not None else ETA_D * np.quantile(data.price, 0.90))
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    load_forecast, pv_forecast, forecast_audit = causal_forecasts(data)

    detail_rows: list[dict] = []
    daily_rows: list[dict] = []
    scenario_audit: list[dict] = []
    validation_rows: list[dict] = []
    emergency_intervals: list[dict] = []
    initial_soc = SOC_INITIAL

    for d in range(1, len(data.dates)):
        date = data.dates[d]
        if d <= 30:
            initial_soc = SOC_INITIAL
        terminal_soc = SOC_INITIAL if d <= 29 else None
        load_scenarios, pv_scenarios, probabilities, sampled = generate_scenarios(
            data, d, load_forecast, pv_forecast
        )
        solution = solve_daily_milp(
            data.price, load_scenarios, pv_scenarios, probabilities, initial_soc, terminal_soc,
            reserve_target=RESERVE_TARGET if args.variant == "reserve" else None,
            reserve_penalty=reserve_penalty,
        )
        validation = validate_daily_solution(
            solution, data.price, load_scenarios, pv_scenarios, initial_soc, terminal_soc
        )
        if not validation["feasible"]:
            raise RuntimeError(f"{date.date()} 约束校验失败：{validation}")

        actual_emergency = np.maximum(
            data.load_kwh[d] + solution.charge
            - solution.purchase - data.pv_kwh[d] - solution.discharge,
            0.0,
        )
        actual_surplus = np.maximum(
            solution.purchase + data.pv_kwh[d] + solution.discharge
            - data.load_kwh[d] - solution.charge,
            0.0,
        )
        actual_balance = (
            solution.purchase + actual_emergency + data.pv_kwh[d]
            - actual_surplus + solution.discharge - data.load_kwh[d] - solution.charge
        )
        plan_cost = float(data.price @ solution.purchase)
        emergency_cost = float((5.0 * data.price) @ actual_emergency)
        actual_cost = plan_cost + emergency_cost
        date_intervals = merge_emergency_intervals(date, actual_emergency, data.interval_labels)
        emergency_intervals.extend(date_intervals)

        for t in range(N):
            detail_rows.append({
                "日期": date,
                "时段编号": t + 1,
                "时间段": data.interval_labels[t],
                "电价(元/kWh)": data.price[t],
                "实际负荷(kWh)": data.load_kwh[d, t],
                "预测负荷(kWh)": load_forecast[d, t],
                "实际光伏(kWh)": data.pv_kwh[d, t],
                "预测光伏(kWh)": pv_forecast[d, t],
                "计划购电量(kWh)": solution.purchase[t],
                "充电量(kWh)": solution.charge[t],
                "放电量(kWh)": solution.discharge[t],
                "时段初储电量(kWh)": initial_soc if t == 0 else solution.soc[t - 1],
                "时段末储电量(kWh)": solution.soc[t],
                "实际紧急购电量(kWh)": actual_emergency[t],
                "实际富余电量(kWh)": actual_surplus[t],
                "正常购电费(元)": data.price[t] * solution.purchase[t],
                "紧急购电费(元)": 5.0 * data.price[t] * actual_emergency[t],
                "实际电量平衡残差(kWh)": actual_balance[t],
                "充放电状态": solution.mode[t],
            })

        daily_rows.append({
            "日期": date,
            "日初储电量(kWh)": initial_soc,
            "日末储电量(kWh)": solution.soc[-1],
            "预测负荷量(kWh)": load_forecast[d].sum(),
            "实际负荷量(kWh)": data.load_kwh[d].sum(),
            "预测光伏量(kWh)": pv_forecast[d].sum(),
            "实际光伏量(kWh)": data.pv_kwh[d].sum(),
            "计划购电量(kWh)": solution.purchase.sum(),
            "实际紧急购电量(kWh)": actual_emergency.sum(),
            "实际富余电量(kWh)": actual_surplus.sum(),
            "充电量(kWh)": solution.charge.sum(),
            "放电量(kWh)": solution.discharge.sum(),
            "正常购电费(元)": plan_cost,
            "紧急购电费(元)": emergency_cost,
            "实际总费用(元)": actual_cost,
            "期望紧急购电费(元)": solution.expected_emergency_cost_yuan,
            "日备不足量(kWh)": solution.reserve_shortfall_kwh,
            "日备不足惩罚(元)": solution.reserve_penalty_yuan,
            "MILP目标值(元)": solution.objective_yuan,
            "场景数": len(probabilities),
            "紧急购电区间数": len(date_intervals),
            "求解耗时(s)": solution.solve_seconds,
            "MIP相对间隙": solution.solver_details.get("mip_gap"),
        })
        scenario_audit.append({
            "日期": date.strftime("%Y-%m-%d"),
            "随机种子": int(date.strftime("%Y%m%d")),
            "场景数": len(probabilities),
            "残差来源日期": [data.dates[i].strftime("%Y-%m-%d") for i in sampled],
        })
        validation_rows.append({"日期": date.strftime("%Y-%m-%d"), **validation,
            "max_abs_actual_balance_residual_kwh": float(np.max(np.abs(actual_balance)))})
        initial_soc = float(solution.soc[-1])
        if d % 25 == 0 or d == len(data.dates) - 1:
            print(f"已完成 {date.date()}，目标值={solution.objective_yuan:.2f}，耗时={solution.solve_seconds:.2f}s")

    detail = pd.DataFrame(detail_rows)
    daily = pd.DataFrame(daily_rows)
    official_detail = detail[(detail["日期"] >= OFFICIAL_START) & (detail["日期"] <= OFFICIAL_END)].copy()
    official_daily = daily[(daily["日期"] >= OFFICIAL_START) & (daily["日期"] <= OFFICIAL_END)].copy()
    official_emergency = [row for row in emergency_intervals if OFFICIAL_START <= pd.Timestamp(row["日期"]) <= OFFICIAL_END]

    if args.variant == "no-reserve":
        detail_path = INTERMEDIATE_DIR / "问题二_无日备惩罚_逐时段结果.csv"
        daily_path = INTERMEDIATE_DIR / "问题二_无日备惩罚_逐日汇总.csv"
    else:
        detail_path = INTERMEDIATE_DIR / "问题二_逐时段结果.csv"
        daily_path = INTERMEDIATE_DIR / "问题二_逐日汇总.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(forecast_audit).to_csv(INTERMEDIATE_DIR / "问题二_预测信息边界审计.csv", index=False, encoding="utf-8-sig")
    write_json(INTERMEDIATE_DIR / "问题二_场景抽样审计.json", scenario_audit)

    official_slice = slice(31, len(data.dates))
    metrics = {
        "负荷": prediction_metrics(data.load_kwh[official_slice], load_forecast[official_slice]),
        "光伏_全时段": prediction_metrics(data.pv_kwh[official_slice], pv_forecast[official_slice]),
        "光伏_实际非零时段": prediction_metrics(data.pv_kwh[official_slice], pv_forecast[official_slice], pv=True),
    }
    max_validation = {
        "all_days_feasible": all(row["feasible"] for row in validation_rows),
        "max_abs_scenario_balance_residual_kwh": max(row["max_abs_scenario_balance_residual_kwh"] for row in validation_rows),
        "max_abs_actual_balance_residual_kwh": max(row["max_abs_actual_balance_residual_kwh"] for row in validation_rows),
        "max_abs_soc_recursion_residual_kwh": max(row["max_abs_soc_recursion_residual_kwh"] for row in validation_rows),
        "simultaneous_charge_discharge_count": sum(row["simultaneous_charge_discharge_count"] for row in validation_rows),
        "min_soc_kwh": min(row["min_soc_kwh"] for row in validation_rows),
        "max_soc_kwh": max(row["max_soc_kwh"] for row in validation_rows),
        "max_charge_kwh": max(row["max_charge_kwh"] for row in validation_rows),
        "max_discharge_kwh": max(row["max_discharge_kwh"] for row in validation_rows),
        "max_mip_gap": max((row["mip_gap"] or 0.0) for row in validation_rows),
    }
    summary = {
        "正式区间": f"{OFFICIAL_START.date()}至{OFFICIAL_END.date()}",
        "正式天数": int(len(official_daily)),
        "周期参数": {"负荷周周期数": 4, "光伏日周期数": 7, "rho": RHO},
        "目标函数参数": {
            "场景数": SCENARIO_COUNT,
            "CVaR风险项": "已取消",
            "富余电量极小惩罚": 0.0,
            "日备目标_kWh": RESERVE_TARGET if args.variant == "reserve" else None,
            "日备不足惩罚系数_元每kWh": reserve_penalty,
        },
        "预测指标": metrics,
        "计划购电量_kWh": float(official_daily["计划购电量(kWh)"].sum()),
        "实际紧急购电量_kWh": float(official_daily["实际紧急购电量(kWh)"].sum()),
        "实际富余电量_kWh": float(official_daily["实际富余电量(kWh)"].sum()),
        "充电量_kWh": float(official_daily["充电量(kWh)"].sum()),
        "放电量_kWh": float(official_daily["放电量(kWh)"].sum()),
        "正常购电费_元": float(official_daily["正常购电费(元)"].sum()),
        "紧急购电费_元": float(official_daily["紧急购电费(元)"].sum()),
        "实际总费用_元": float(official_daily["实际总费用(元)"].sum()),
        "日备不足量_kWh": float(official_daily["日备不足量(kWh)"].sum()),
        "日备不足惩罚_元": float(official_daily["日备不足惩罚(元)"].sum()),
        "MILP目标值_元": float(official_daily["MILP目标值(元)"].sum()),
        "平均日末储电量_kWh": float(official_daily["日末储电量(kWh)"].mean()),
        "发生紧急购电的时段数": int((official_detail["实际紧急购电量(kWh)"] > CHECK_TOL).sum()),
        "2月1日初始储电量_kWh": float(official_daily.iloc[0]["日初储电量(kWh)"]),
        "12月31日末储电量_kWh": float(official_daily.iloc[-1]["日末储电量(kWh)"]),
        "平均单日求解耗时_s": float(daily["求解耗时(s)"].mean()),
        "最大单日求解耗时_s": float(daily["求解耗时(s)"].max()),
        "约束校验": max_validation,
    }
    summary_path = (
        INTERMEDIATE_DIR / "问题二_无日备惩罚_汇总与校验.json"
        if args.variant == "no-reserve" else INTERMEDIATE_DIR / "问题二_汇总与校验.json"
    )
    write_json(summary_path, summary)

    if args.variant == "no-reserve":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    baseline_daily_path = INTERMEDIATE_DIR / "问题二_无日备惩罚_逐日汇总.csv"
    if not baseline_daily_path.exists():
        raise FileNotFoundError("缺少无日备惩罚对照结果，请先运行 --variant no-reserve")
    baseline_daily = pd.read_csv(baseline_daily_path, parse_dates=["日期"])
    comparison = build_comparison(daily, baseline_daily)
    summary["方案对比"] = comparison
    write_json(summary_path, summary)

    purchase_rows = []
    storage_rows = []
    for _, row in official_daily.iterrows():
        date = pd.Timestamp(row["日期"])
        day_detail = official_detail[official_detail["日期"] == date]
        purchase_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "purchase": day_detail["计划购电量(kWh)"].tolist(),
            "total_purchase": float(row["计划购电量(kWh)"]),
            "total_cost": float(row["实际总费用(元)"]),
        })
        storage_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "charge_4h": [float(day_detail["充电量(kWh)"].iloc[i:i+24].sum()) for i in range(0, N, 24)],
            "discharge_4h": [float(day_detail["放电量(kWh)"].iloc[i:i+24].sum()) for i in range(0, N, 24)],
            "initial_soc": float(row["日初储电量(kWh)"]),
            "terminal_soc": float(row["日末储电量(kWh)"]),
        })
    workbook_bundle = {
        "purchaseRows": purchase_rows,
        "storageRows": storage_rows,
        "emergencyRows": official_emergency,
        "summary": summary,
        "comparison": comparison,
    }
    write_json(INTERMEDIATE_DIR / "问题二_工作簿数据.json", workbook_bundle)

    # 预测与周期图不受目标函数变化影响；最终方案只重绘受调度变化影响的两张图。
    figure_paths = [
        draw_yearly_figure(detail),
        draw_comparison_figure(daily, baseline_daily),
    ]
    summary["论文图"] = [str(path.relative_to(ROOT)) for path in figure_paths]
    write_json(INTERMEDIATE_DIR / "问题二_汇总与校验.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
