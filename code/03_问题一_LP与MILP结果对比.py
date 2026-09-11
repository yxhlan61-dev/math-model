"""一次运行无储能、LP 和 MILP 三种方案，生成最终对比所需数据。"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from problem1_solver_common import OUTPUT_DIR, ROOT, check_solution, load_input, solve_lp, solve_milp, write_outputs


def main():
    data = load_input()
    lp_solution = solve_lp(data)
    milp_solution = solve_milp(data)
    _, lp_check = write_outputs(data, lp_solution)
    _, milp_check = write_outputs(data, milp_solution)

    no_storage_purchase = np.maximum(data.load_kwh - data.pv_kwh, 0.0)
    no_storage_curtailment = np.maximum(data.pv_kwh - data.load_kwh, 0.0)
    no_storage_balance_residual = no_storage_purchase + data.pv_kwh - no_storage_curtailment - data.load_kwh
    no_storage_cost = float(data.price @ no_storage_purchase)
    zero_six = [0.0] * 6
    no_storage_bundle = {
        "method": "无储能基准",
        "submission": {
            "purchase": no_storage_purchase.tolist(),
            "fourHourCharge": zero_six,
            "fourHourDischarge": zero_six,
            "initialSoc": 0.0,
            "terminalSoc": 0.0,
        },
        "checks": {
            "feasible": bool(np.max(np.abs(no_storage_balance_residual)) <= 1e-6),
            "max_abs_power_balance_residual_kwh": float(np.max(np.abs(no_storage_balance_residual))),
            "objective_yuan": no_storage_cost,
            "total_purchase_kwh": float(no_storage_purchase.sum()),
            "total_pv_curtailment_kwh": float(no_storage_curtailment.sum()),
        },
    }
    no_storage_path = OUTPUT_DIR / "问题一_无储能基准_工作簿数据.json"
    no_storage_path.write_text(json.dumps(no_storage_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    comparison = pd.DataFrame([
        {
            "方法": "无储能基准",
            "购电费用(元)": no_storage_cost,
            "较无储能节省费用(元)": 0.0,
            "较无储能节省率": 0.0,
            "购电量(kWh)": no_storage_purchase.sum(),
            "充电量(kWh)": 0.0,
            "放电量(kWh)": 0.0,
            "弃光量(kWh)": no_storage_curtailment.sum(),
            "同时充放电时段数": 0,
            "最大平衡残差(kWh)": np.max(np.abs(no_storage_balance_residual)),
            "求解耗时(s)": 0.0,
        },
        {
            "方法": "LP（事后核查）",
            "购电费用(元)": lp_solution.objective_yuan,
            "较无储能节省费用(元)": no_storage_cost - lp_solution.objective_yuan,
            "较无储能节省率": (no_storage_cost - lp_solution.objective_yuan) / no_storage_cost,
            "购电量(kWh)": lp_solution.purchase.sum(),
            "充电量(kWh)": lp_solution.charge.sum(),
            "放电量(kWh)": lp_solution.discharge.sum(),
            "弃光量(kWh)": lp_solution.curtailment.sum(),
            "同时充放电时段数": lp_check["simultaneous_charge_discharge_count"],
            "最大平衡残差(kWh)": lp_check["max_abs_power_balance_residual_kwh"],
            "求解耗时(s)": lp_solution.solve_seconds,
        },
        {
            "方法": "MILP（显式互斥）",
            "购电费用(元)": milp_solution.objective_yuan,
            "较无储能节省费用(元)": no_storage_cost - milp_solution.objective_yuan,
            "较无储能节省率": (no_storage_cost - milp_solution.objective_yuan) / no_storage_cost,
            "购电量(kWh)": milp_solution.purchase.sum(),
            "充电量(kWh)": milp_solution.charge.sum(),
            "放电量(kWh)": milp_solution.discharge.sum(),
            "弃光量(kWh)": milp_solution.curtailment.sum(),
            "同时充放电时段数": milp_check["simultaneous_charge_discharge_count"],
            "最大平衡残差(kWh)": milp_check["max_abs_power_balance_residual_kwh"],
            "求解耗时(s)": milp_solution.solve_seconds,
        },
    ])
    lp_milp_difference = pd.DataFrame([
        {"对比指标": "购电费用差", "数值": milp_solution.objective_yuan - lp_solution.objective_yuan, "单位": "元"},
        {"对比指标": "购电量最大逐时段差", "数值": abs(milp_solution.purchase - lp_solution.purchase).max(), "单位": "kWh"},
        {"对比指标": "储电量最大逐时段差", "数值": abs(milp_solution.soc - lp_solution.soc).max(), "单位": "kWh"},
        {"对比指标": "购电计划差异时段数", "数值": int((abs(milp_solution.purchase - lp_solution.purchase) > 1e-6).sum()), "单位": "个"},
    ])
    comparison_json = OUTPUT_DIR / "问题一_三种方案结果对比.json"
    comparison_json.write_text(
        json.dumps({
            "columns": comparison.columns.tolist(),
            "rows": comparison.values.tolist(),
            "differenceColumns": lp_milp_difference.columns.tolist(),
            "differenceRows": lp_milp_difference.values.tolist(),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figures = ROOT / "figures" / "问题一"
    figures.mkdir(parents=True, exist_ok=True)
    # 功率和电价是10分钟区间平均值，横坐标使用区间中点；SOC另外补入0:00初值。
    interval_hours = (data.interval_id - 0.5) / 6
    soc_hours = np.arange(0, len(data.interval_id) + 1) / 6
    # 与问题二、三、四的 24 小时时序图统一：0--24 h，每 3 h 一个主刻度。
    tick_hours = np.arange(0, 25, 3)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=False, constrained_layout=True)

    purchase_ax = axes[0]
    purchase_ax.plot(interval_hours, data.load_kw, color="#D62728", label="负荷", linewidth=1.25)
    purchase_ax.plot(interval_hours, data.pv_kw, color="#2CA02C", label="光伏", linewidth=1.25)
    purchase_ax.plot(
        interval_hours,
        no_storage_purchase * 6,
        color="#7B2CBF",
        linestyle="-.",
        label="无储能购电",
        linewidth=1.8,
        zorder=4,
    )
    purchase_ax.plot(interval_hours, lp_solution.purchase * 6, color="#FF7F0E", label="LP购电", linewidth=1.1)
    purchase_ax.plot(interval_hours, milp_solution.purchase * 6, color="#1F77B4", linestyle="--", label="MILP购电", linewidth=1.1)
    purchase_ax.set_xlim(0, 24)
    purchase_ax.set_ylim(0, 10000)
    purchase_ax.set_xticks(tick_hours)
    purchase_ax.set_xlabel("时刻/h")
    purchase_ax.set_ylabel("功率/kW")
    purchase_ax.grid(True, which="major", color="#C8C8C8", linestyle="--", linewidth=0.7, alpha=0.8)
    purchase_ax.set_axisbelow(True)
    purchase_ax.legend(ncol=5, fontsize=8, loc="upper center")

    storage_ax = axes[1]
    storage_ax.plot(soc_hours, np.r_[6000.0, lp_solution.soc], color="#FF7F0E", label="LP储电量", linewidth=1.35)
    storage_ax.plot(soc_hours, np.r_[6000.0, milp_solution.soc], color="#1F77B4", linestyle="--", label="MILP储电量", linewidth=1.35)
    storage_ax.axhline(1200, color="#707070", linewidth=0.75, label="储电量上下限")
    storage_ax.axhline(10800, color="#707070", linewidth=0.75)
    storage_ax.set_xlim(0, 24)
    storage_ax.set_ylim(0, 12500)
    storage_ax.set_xticks(tick_hours)
    storage_ax.set_xlabel("时刻/h")
    storage_ax.set_ylabel("储电量/kWh")
    storage_ax.grid(True, which="major", color="#C8C8C8", linestyle="--", linewidth=0.7, alpha=0.8)
    storage_ax.set_axisbelow(True)

    price_ax = storage_ax.twinx()
    price_ax.plot(interval_hours, data.price, color="#B2182B", label="电价", linewidth=1.25, alpha=0.95)
    price_ax.set_ylim(0.3, 1.55)
    price_ax.set_ylabel("电价/(元/kWh)", color="#B2182B")
    price_ax.tick_params(axis="y", colors="#B2182B")

    storage_lines, storage_labels = storage_ax.get_legend_handles_labels()
    price_lines, price_labels = price_ax.get_legend_handles_labels()
    storage_ax.legend(storage_lines + price_lines, storage_labels + price_labels, ncol=4, fontsize=8, loc="upper center")
    fig.savefig(figures / "问题一_三种方案调度结果对比.pdf", bbox_inches="tight")
    plt.close(fig)
    print(comparison.to_string(index=False))
    print(f"对比数据：{comparison_json}")


if __name__ == "__main__":
    main()
