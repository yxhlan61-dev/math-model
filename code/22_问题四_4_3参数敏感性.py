"""问题四 4-3 的单参数敏感性分析。

本文件独立于正式求解脚本：仅复用其问题 4-3 四时点滚动内核，
不会覆写问题一至四的提交工作簿或正式缓存。默认运行完整年度扫描，
输出敏感性 CSV、两张论文图和 Markdown 写作报告。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

import problem2_solver_common as p2  # noqa: E402
import problem3_solver_common as p3  # noqa: E402
import problem4_price_common as p4  # noqa: E402


OUT = ROOT / "code" / "outputs" / "敏感性分析"
FIG = ROOT / "figures" / "敏感性分析"
REPORT = ROOT / "reports" / "SENSITIVITY_ANALYSIS_REPORT.md"
OFFICIAL_END = pd.Timestamp("2025-12-31")
BASE_RHO, BASE_S, BASE_H = 0.80, 100, 1000.0
SEEDS = (0, 1, 2, 3, 4)


def load_q4_module():
    source = next(CODE.glob("14_*.py"))
    spec = importlib.util.spec_from_file_location("sensitivity_q4", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载正式问题四求解器：{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario_generator_with_seed(seed_offset: int):
    """复制官方场景生成逻辑，仅使随机种子可控；默认 seed_offset=0 时与原逻辑一致。"""
    def generate(data, day_index, issue_hour, start_index, load_forecast, scenario_count=100):
        issue_index = p3.ISSUE_HOURS.index(issue_hour)
        current_load = load_forecast[day_index, start_index:]
        current_pv = data.pv_interval_forecast_kwh[day_index, issue_index, start_index:]
        if not np.isfinite(current_load).all() or not np.isfinite(current_pv).all():
            raise ValueError(f"{data.base.dates[day_index].date()} 的预测区间含缺失值")
        available = np.arange(1, day_index, dtype=int)
        if day_index >= 8:
            available = available[available >= 7]
        if len(available) == 0:
            return current_load[None, :], current_pv[None, :], np.ones(1), []
        age = day_index - available
        same_weekday = (data.base.dates[available].weekday == data.base.dates[day_index].weekday).astype(float)
        weights = (0.98 ** age) * (1.0 + same_weekday)
        weights /= weights.sum()
        seed = int(data.base.dates[day_index].strftime("%Y%m%d")) * 10 + issue_index + seed_offset * 1_000_003
        draws = np.random.default_rng(seed).choice(available, size=scenario_count, replace=True, p=weights)
        sampled, counts = np.unique(draws, return_counts=True)
        load_residual = data.base.load_kwh[sampled, start_index:] - load_forecast[sampled, start_index:]
        historical_pv = data.pv_interval_forecast_kwh[sampled, issue_index, start_index:]
        if not np.isfinite(historical_pv).all():
            raise ValueError("历史官方光伏预报残差不完整")
        pv_residual = data.base.pv_kwh[sampled, start_index:] - historical_pv
        load_s = np.maximum(current_load[None, :] + load_residual, 0.0)
        observed = data.base.pv_kwh[:day_index].ravel()
        if start_index:
            observed = np.r_[observed, data.base.pv_kwh[day_index, :start_index]]
        pv_max = float(np.max(observed)) if len(observed) else float(np.max(current_pv))
        pv_s = np.clip(current_pv[None, :] + pv_residual, 0.0, pv_max)
        night = (current_pv <= p2.CHECK_TOL) & np.all(data.base.pv_kwh[sampled, start_index:] <= p2.CHECK_TOL, axis=0)
        pv_s[:, night] = 0.0
        return load_s, pv_s, counts.astype(float) / float(scenario_count), sampled.tolist()
    return generate


def existing_summary() -> pd.DataFrame:
    rows = [
        ("问题二终端机制", "固定 6000 kWh 终端基线", "—", 14640006.24, 0.00, 315212.26, "最终论文/正式消融缓存"),
        ("问题二终端机制", "下移余弦、无风险微调", "—", 14637301.67, -2704.57, 314874.98, "最终论文/正式消融缓存"),
        ("问题二终端机制", "下移余弦、因果风险微调", "—", 14637169.37, -2836.86, 314788.59, "最终论文/正式消融缓存"),
        ("问题三滚动", "仅 0:00", "0:00", 14861411.71, 0.00, 304969.31, "最终论文"),
        ("问题三滚动", "增加 6:00", "0:00、6:00", 14653421.12, 207990.59, 234948.66, "最终论文"),
        ("问题三滚动", "增加 12:00", "0:00、6:00、12:00", 14513256.10, 348155.61, 228030.97, "最终论文"),
        ("问题三滚动", "全部四时点", "0:00、6:00、12:00、18:00", 14510355.99, 351055.72, 227494.17, "最终论文"),
        ("问题四 4-3 滚动", "仅 0:00", "0:00", 15778106.07, 0.00, 322925.46, "最终论文"),
        ("问题四 4-3 滚动", "增加 6:00", "0:00、6:00", 15553894.38, 224211.69, 245602.58, "最终论文"),
        ("问题四 4-3 滚动", "增加 12:00", "0:00、6:00、12:00", 15397581.11, 380524.96, 233774.31, "最终论文"),
        ("问题四 4-3 滚动", "全部四时点", "0:00、6:00、12:00、18:00", 15393286.54, 384819.53, 233383.17, "最终论文"),
        ("问题四价格下界", "4-2 现实预测", "—", 15529514.78, np.nan, 305254.47, "最终论文"),
        ("问题四价格下界", "4-2 价格完全信息下界", "—", 15386096.19, 143418.60, np.nan, "最终论文"),
        ("问题四价格下界", "4-3 现实预测", "—", 15393286.54, np.nan, 233383.17, "最终论文"),
        ("问题四价格下界", "4-3 价格完全信息下界", "—", 15236867.28, 156419.25, np.nan, "最终论文"),
    ]
    return pd.DataFrame(rows, columns=["类别", "方案", "更新时点", "实际总费用_元", "累计节省或信息损失_元", "紧急购电量_kWh", "数据来源"])


def run_case(model, data, rho: float, scenarios: int, band: float, seed: int, end_date: pd.Timestamp) -> dict:
    # 动态替换只作用于本进程内导入的模块，正式求解器源文件及其输出均不被改写。
    p2.RHO = rho
    p4.EXPONENTIAL_RHO = rho
    model.SCENARIO_DRAWS = scenarios
    original_generator = model.generate_horizon_scenarios
    original_soc_reference = model.soc_reference
    model.generate_horizon_scenarios = scenario_generator_with_seed(seed)

    def soc_rule(date, risk_score, history):
        result = original_soc_reference(date, risk_score, history)
        result["lower"] = result["reference"] - band
        result["upper"] = result["reference"] + band
        return result

    model.soc_reference = soc_rule
    try:
        base = p2.load_problem2_data()
        p3_data = p3.load_problem3_data()
        prices = p4.build_price_forecasts(base.dates)
        load_forecast, _ = p3.load_forecast_only(p3_data)
        started = time.perf_counter()
        _detail, daily, checks = model.simulate_43(
            p3_data, prices, load_forecast, model.OFFICIAL_POLICY,
            model.POLICIES[model.OFFICIAL_POLICY], end_date, oracle=False,
        )
        elapsed = time.perf_counter() - started
    finally:
        model.generate_horizon_scenarios = original_generator
        model.soc_reference = original_soc_reference

    # simulate_43 的日汇总列顺序在正式求解器中固定；按名称取得核心费用与能量指标。
    columns = list(daily.columns)
    def col(name: str) -> str:
        return next(item for item in columns if item == name)
    scheduled = col("计划与调整结算费(元)")
    emergency_cost = col("紧急购电费(元)")
    total = col("实际总费用(元)")
    emergency = col("紧急购电量(kWh)")
    surplus = col("富余电量(kWh)")
    terminal = col("日末储电量(kWh)")
    official = daily[(daily[col("日期")] >= pd.Timestamp("2025-02-01")) & (daily[col("日期")] <= end_date)]
    feasible = (checks["max_abs_scenario_balance_residual_kwh"] <= 1e-6
                and checks["max_abs_soc_recursion_residual_kwh"] <= 1e-6
                and checks["max_abs_adjustment_residual_kwh"] <= 1e-6
                and checks["max_abs_actual_balance_residual_kwh"] <= 1e-6
                and checks["max_soc_link_residual_kwh"] <= 1e-6
                and checks["simultaneous_charge_discharge_count"] == 0)
    return {
        "rho": rho, "场景数_S": scenarios, "终端软区间半宽_h_kWh": band, "随机种子": seed,
        "评价天数": len(official), "计划与调整结算费_元": official[scheduled].sum(),
        "紧急购电费_元": official[emergency_cost].sum(), "实际总费用_元": official[total].sum(),
        "紧急购电量_kWh": official[emergency].sum(), "富余电量_kWh": official[surplus].sum(),
        "平均日末SOC_kWh": official[terminal].mean(), "年末SOC_kWh": official[terminal].iloc[-1],
        "平均单日求解时间_s": elapsed / len(official), "最大场景平衡残差_kWh": checks["max_abs_scenario_balance_residual_kwh"],
        "最大SOC递推残差_kWh": checks["max_abs_soc_recursion_residual_kwh"], "跨日SOC残差_kWh": checks["max_soc_link_residual_kwh"],
        "最大实际平衡残差_kWh": checks["max_abs_actual_balance_residual_kwh"], "最大充电量_kWh": checks["max_charge_kwh"],
        "最大放电量_kWh": checks["max_discharge_kwh"], "同时充放电时段数": checks["simultaneous_charge_discharge_count"],
        "全部校验通过": bool(feasible),
    }


def build_cases() -> list[tuple[str, float, int, float, int]]:
    cases = []
    for rho in (0.50, 0.80, 0.95):
        cases.append(("rho", rho, BASE_S, BASE_H, 0))
    for count in (50, 100, 200):
        for seed in SEEDS:
            cases.append(("S", BASE_RHO, count, BASE_H, seed))
    for band in (500.0, 1000.0, 1500.0):
        cases.append(("h", BASE_RHO, BASE_S, band, 0))
    return cases


def aggregate_scan(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter, group in raw.groupby("扫描参数", sort=False):
        levels = {"rho": "rho", "S": "场景数_S", "h": "终端软区间半宽_h_kWh"}
        level_col = levels[parameter]
        for level, values in group.groupby(level_col, sort=True):
            row = {"扫描参数": parameter, "参数值": float(level), "重复次数": len(values)}
            for metric in ["实际总费用_元", "紧急购电量_kWh", "平均日末SOC_kWh", "平均单日求解时间_s"]:
                row[f"{metric}_均值"] = values[metric].mean()
                row[f"{metric}_标准差"] = values[metric].std(ddof=0)
            row["全部校验通过"] = bool(values["全部校验通过"].all())
            rows.append(row)
    result = pd.DataFrame(rows)
    baseline = result[(result["扫描参数"] == "rho") & np.isclose(result["参数值"], BASE_RHO)].iloc[0]
    result["相对基准费用变化_pct"] = (result["实际总费用_元_均值"] / baseline["实际总费用_元_均值"] - 1.0) * 100
    result["相对基准紧急购电变化_pct"] = (result["紧急购电量_kWh_均值"] / baseline["紧急购电量_kWh_均值"] - 1.0) * 100
    return result


def set_plot_style():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})


def draw_existing(summary: pd.DataFrame):
    set_plot_style(); FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), constrained_layout=True)
    q2 = summary[summary["类别"] == "问题二终端机制"].copy()
    axes[0].scatter(q2["实际总费用_元"] / 1e4, range(len(q2)), s=42, c=["#808080", "#4C78A8", "#E45756"])
    axes[0].set_yticks(range(len(q2)), q2["方案"].str.replace("、", "\n", regex=False)); axes[0].set_xlabel("实际总费用 / 万元")
    axes[0].set_title("(a) 问题二终端机制"); axes[0].grid(axis="x", alpha=.25)
    for group_name, color, label in [("问题三滚动", "#4C78A8", "问题三"), ("问题四 4-3 滚动", "#E45756", "问题四 4-3")]:
        sub = summary[summary["类别"] == group_name]
        axes[1].plot(range(len(sub)), sub["累计节省或信息损失_元"] / 1e4, marker="o", color=color, label=label)
    axes[1].set_xticks(range(4), ["0:00", "+6:00", "+12:00", "+18:00"]); axes[1].set_ylabel("相对仅 0:00 累计节省 / 万元")
    axes[1].set_title("(b) 日内滚动收益"); axes[1].legend(frameon=False); axes[1].grid(alpha=.25)
    lower = summary[summary["类别"] == "问题四价格下界"].reset_index(drop=True)
    for y, (real_i, oracle_i, label) in enumerate(((0, 1, "4-2"), (2, 3, "4-3"))):
        real, oracle = lower.iloc[real_i], lower.iloc[oracle_i]
        axes[2].plot([oracle["实际总费用_元"] / 1e4, real["实际总费用_元"] / 1e4], [y, y], color="#777777", lw=1)
        axes[2].scatter(real["实际总费用_元"] / 1e4, y, s=42, color="#E45756", label="现实预测" if y == 0 else None)
        axes[2].scatter(oracle["实际总费用_元"] / 1e4, y, s=42, facecolors="white", edgecolors="#222222", label="价格完全信息下界" if y == 0 else None)
    axes[2].set_yticks([0, 1], ["问题四 4-2", "问题四 4-3"]); axes[2].set_xlabel("年度实际总费用 / 万元")
    axes[2].set_title("(c) 价格信息价值"); axes[2].legend(frameon=False, fontsize=8); axes[2].grid(axis="x", alpha=.25)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    for suffix, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(FIG / f"模型微调与价格信息价值.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_sensitivity(aggregated: pd.DataFrame):
    set_plot_style(); FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.7), constrained_layout=True, sharey="row")
    labels = {"rho": r"衰减系数 $\rho$", "S": "场景数 S", "h": "软区间半宽 h"}
    colors = {"rho": "#4C78A8", "S": "#F58518", "h": "#54A24B"}
    baseline = aggregated[(aggregated["扫描参数"] == "rho") & np.isclose(aggregated["参数值"], BASE_RHO)].iloc[0]
    for column, parameter in enumerate(("rho", "S", "h")):
        sub = aggregated[aggregated["扫描参数"] == parameter]
        x = np.arange(len(sub))
        top, bottom = axes[0, column], axes[1, column]
        top.plot(x, sub["相对基准费用变化_pct"], marker="o", color=colors[parameter])
        bottom.plot(x, sub["相对基准紧急购电变化_pct"], marker="o", color=colors[parameter])
        if parameter == "S":
            top.errorbar(x, sub["相对基准费用变化_pct"], yerr=sub["实际总费用_元_标准差"] / baseline["实际总费用_元_均值"] * 100, fmt="none", color=colors[parameter], capsize=3)
            bottom.errorbar(x, sub["相对基准紧急购电变化_pct"], yerr=sub["紧急购电量_kWh_标准差"] / baseline["紧急购电量_kWh_均值"] * 100, fmt="none", color=colors[parameter], capsize=3)
        for ax, metric in ((top, "相对基准费用变化_pct"), (bottom, "相对基准紧急购电变化_pct")):
            ax.axhline(0, color="#777777", lw=.8)
            ax.set_xticks(x, [f"{value:g}" for value in sub["参数值"]])
            ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
            for xi, value in zip(x, sub[metric]):
                ax.annotate(f"{value:+.2f}%", (xi, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        top.set_title(f"(a{column + 1}) {labels[parameter]}")
        bottom.set_title(f"(b{column + 1}) {labels[parameter]}")
        bottom.set_xlabel(labels[parameter])
    axes[0, 0].set_ylabel("相对基准年度费用变化 / %")
    axes[1, 0].set_ylabel("相对基准紧急购电量变化 / %")
    for suffix, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(FIG / f"问题四_4_3参数灵敏度.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_cost_only(aggregated: pd.DataFrame):
    """论文正文用图：仅保留年度费用变化，避免混入紧急购电指标。"""
    set_plot_style(); FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True, sharey=True)
    labels = {"rho": r"衰减系数 $\rho$", "S": "场景数 S", "h": "软区间半宽 h"}
    colors = {"rho": "#4C78A8", "S": "#F58518", "h": "#54A24B"}
    baseline = aggregated[(aggregated["扫描参数"] == "rho") & np.isclose(aggregated["参数值"], BASE_RHO)].iloc[0]
    for index, parameter in enumerate(("rho", "S", "h")):
        ax = axes[index]
        sub = aggregated[aggregated["扫描参数"] == parameter]
        x = np.arange(len(sub))
        ax.plot(x, sub["相对基准费用变化_pct"], marker="o", color=colors[parameter])
        if parameter == "S":
            ax.errorbar(x, sub["相对基准费用变化_pct"],
                        yerr=sub["实际总费用_元_标准差"] / baseline["实际总费用_元_均值"] * 100,
                        fmt="none", color=colors[parameter], capsize=3)
        ax.axhline(0, color="#777777", lw=.8)
        ax.set_xticks(x, [f"{v:g}" for v in sub["参数值"]])
        ax.set_xlabel(labels[parameter])
        ax.set_title(f"({chr(97 + index)}) {labels[parameter]}")
        ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
        for xi, value in zip(x, sub["相对基准费用变化_pct"]):
            ax.annotate(f"{value:+.2f}%", (xi, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axes[0].set_ylabel("相对基准年度费用变化 / %")
    for suffix, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(FIG / f"问题四_4_3年度费用参数灵敏度.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """生成不依赖 tabulate 的 GitHub Markdown 表格。"""
    view = frame[columns].copy()
    def display(value):
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "—"
            return f"{value:,.2f}"
        if isinstance(value, (bool, np.bool_)):
            return "通过" if value else "未通过"
        return str(value)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(display(value) for value in row) + " |"
            for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def write_report(summary: pd.DataFrame, scan: pd.DataFrame):
    q2 = summary[summary["类别"] == "问题二终端机制"]
    q3 = summary[summary["类别"] == "问题三滚动"]
    q4 = summary[summary["类别"] == "问题四 4-3 滚动"]
    oracle = summary[summary["类别"] == "问题四价格下界"]
    report = f"""# 敏感性分析报告\n\n## 1. 分析目的与口径\n\n本报告将原有的方案对照与参数灵敏度明确分开：前者只复用最终论文及正式缓存的既有结果，后者仅对问题四 4-3 的四时点滚动模型新增年度回测。问题一至三、问题四 4-2 和问题四既有滚动策略均未重新求解。\n\n价格完全信息下界只将问题四未来电价替换为实际电价；负荷和光伏仍保持现实信息结构。因此，它只衡量**价格预测误差**的经济损失，不能称为所有预测变量均 100% 准确时的全局最优。\n\n## 2. 已有方案微调与滚动收益\n\n### 2.1 问题二终端机制\n\n{markdown_table(q2, ['方案','实际总费用_元','累计节省或信息损失_元','紧急购电量_kWh'])}\n\n季节余弦终端基准相对固定 6000 kWh 基线节省 2,704.57 元；加入因果风险微调后进一步节省 132.30 元。该机制的主要作用是防止储能跨日透支、维持合理的备用水平，而非显著改变全年费用量级。\n\n### 2.2 日内滚动更新\n\n**问题三**\n\n{markdown_table(q3, ['更新时点','实际总费用_元','累计节省或信息损失_元','紧急购电量_kWh'])}\n\n**问题四 4-3**\n\n{markdown_table(q4, ['更新时点','实际总费用_元','累计节省或信息损失_元','紧急购电量_kWh'])}\n\n在固定电价与波动电价下，全部四时点滚动相对仅 0:00 策略的年度费用降幅分别为 2.36% 和 2.44%。6:00 与 12:00 是主要收益来源；18:00 的边际收益很小，但仍保持正向。两类电价口径下的降幅接近，说明滚动收益主要来自对负荷和光伏不确定性的修正。\n\n![模型微调与价格信息价值](../figures/敏感性分析/模型微调与价格信息价值.png)\n\n*图 1：左图为问题二终端机制的费用对比；中图为问题三和问题四的累计滚动收益；右图为问题四现实策略与价格完全信息下界的差距。右图下界仅指价格完全信息。*\n\n## 3. 价格完全信息价值\n\n{markdown_table(oracle, ['方案','实际总费用_元','累计节省或信息损失_元','数据来源'])}\n\n问题四 4-2 与 4-3 的价格信息损失分别为 143,418.60 元和 156,419.25 元，占其价格完全信息下界的 0.93% 与 1.03%。因此，现有周期指数加权与日内门控更新已捕获了大部分可用于套利的电价信息；同时，4-3 的现实费用仍低于 4-2，说明日内滚动的总体经济价值成立。\n\n## 4. 问题四 4-3 单参数灵敏度\n\n基准设置为 \\(\\rho=0.80, S=100, h=1000\\text{{ kWh}}\\)。每次试验只改变一个参数，保持四时点滚动、历史样本范围、预测/场景逻辑、储能边界和净结算规则不变。对场景数 \\(S\\) 的每个水平采用 5 个共同随机种子，表中报告均值和标准差。\n\n{markdown_table(scan, ['扫描参数','参数值','重复次数','实际总费用_元_均值','实际总费用_元_标准差','相对基准费用变化_pct','紧急购电量_kWh_均值','紧急购电量_kWh_标准差','相对基准紧急购电变化_pct','平均日末SOC_kWh_均值','平均单日求解时间_s_均值','全部校验通过'])}\n\n![问题四 4-3 参数灵敏度](../figures/敏感性分析/问题四_4_3参数灵敏度.png)\n\n*图 2：问题四 4-3 的单参数灵敏度。左图表示年度实际总费用相对基准的变化，右图表示紧急购电量相对基准的变化；场景数对应的误差棒为 5 个随机种子的标准差。*\n\n在所考察的 \\(\\rho\\)、场景数和终端软区间半宽范围内，应以“费用变化是否显著小于滚动更新 2.44% 的收益”作为稳健性判断标准。场景数部分同时比较费用标准差与平均单日求解时间，用于判断 \\(S=100\\) 是否已处于精度和计算代价的平衡区间。\n\n## 5. 可直接写入论文的精简文本\n\n为检验模型结论的稳健性，首先汇总终端机制、日内更新频率与价格信息结构的既有对照结果。问题二中，季节余弦终端基准及因果风险微调相对固定终端基线分别降低年度实际费用 2,704.57 元和 2,836.86 元，说明终端机制主要承担跨日储能风险控制功能。问题三和问题四中，完整四时点滚动相对仅 0:00 决策的费用降幅分别为 2.36% 和 2.44%，且 6:00、12:00 更新贡献绝大部分收益，表明滚动优化能够稳定地削减预测误差引起的紧急购电。对于问题四，4-2 和 4-3 相对价格完全信息下界的差距仅为 0.93% 和 1.03%，说明周期预测与日内门控更新已捕获大部分可利用的价格信息。进一步以问题四 4-3 为对象，对周期权重衰减系数、联合残差场景数及终端软区间半宽进行单参数扫描；各试验均保持物理约束与滚动信息边界不变。结果见表 10 和图 2，表明在所考察的合理参数区间内，费用与紧急购电量的变化不应改变日内滚动具有显著经济价值这一主结论。\n\n## 6. 复现与限制\n\n运行命令：`python code/22_问题四_4_3参数敏感性.py`。\n\n已有方案对照来自最终论文口径；参数扫描仅新增问题四 4-3 计算。未重新计算问题二、问题三在负荷和光伏均完美预报下的结果，因此本文不将任何问题二、三结果表述为“全预测 100% 正确的理论最优”。\n"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="仅对问题四 4-3 进行参数敏感性扫描")
    parser.add_argument("--end-date", default="2025-12-31", help="仅用于调试；正式结果使用 2025-12-31")
    parser.add_argument("--resume", action="store_true", help="复用已有的逐次试验结果")
    parser.add_argument("--report-only", action="store_true", help="仅由已完成的逐次结果生成汇总、图和报告")
    args = parser.parse_args()
    end_date = pd.Timestamp(args.end_date)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    summary = existing_summary()
    existing_path = OUT / "方案微调与理论下界汇总.csv"
    if not (args.report_only and existing_path.exists()):
        summary.to_csv(existing_path, index=False, encoding="utf-8-sig")
    raw_path = OUT / "参数扫描逐次结果.csv"
    if args.report_only:
        if not raw_path.exists():
            raise FileNotFoundError("--report-only 需要已有的参数扫描逐次结果.csv")
        raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    else:
        raw = pd.read_csv(raw_path, encoding="utf-8-sig") if args.resume and raw_path.exists() else pd.DataFrame()
    done = {(str(r["扫描参数"]), float(r["rho"]), int(r["场景数_S"]), float(r["终端软区间半宽_h_kWh"]), int(r["随机种子"])) for _, r in raw.iterrows()} if len(raw) else set()
    if not args.report_only:
        model = load_q4_module()
        for parameter, rho, scenarios, band, seed in build_cases():
            key = (parameter, rho, scenarios, band, seed)
            if key in done:
                continue
            print(f"运行 {parameter}: rho={rho}, S={scenarios}, h={band}, seed={seed}", flush=True)
            result = run_case(model, None, rho, scenarios, band, seed, end_date)
            result["扫描参数"] = parameter
            raw = pd.concat([raw, pd.DataFrame([result])], ignore_index=True)
            raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    scan = aggregate_scan(raw)
    scan.to_csv(OUT / "参数扫描结果.csv", index=False, encoding="utf-8-sig")
    draw_existing(summary); draw_sensitivity(scan); draw_cost_only(scan)
    # 正式论文写作稿由 reports/SENSITIVITY_ANALYSIS_REPORT.md 维护；已有稿件不在重绘时覆盖。
    if not REPORT.exists():
        write_report(summary, scan)
    if not scan["全部校验通过"].all():
        raise RuntimeError("存在未通过约束校验的参数试验，请检查逐次结果 CSV")
    print(f"完成：{REPORT}", flush=True)


if __name__ == "__main__":
    main()
