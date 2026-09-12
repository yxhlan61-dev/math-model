"""问题二优化版：两日 SOC 价值、每 10 分钟储能纠偏与因果预测择优。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from problem2_solver_common import (
    CHECK_TOL, FIGURE_DIR, INTERMEDIATE_DIR, N, OFFICIAL_END, OFFICIAL_START,
    OUTPUT_DIR, ROOT, SOC_INITIAL, load_problem2_data, merge_emergency_intervals,
    prediction_metrics, write_json,
)
from problem2_recourse_common import (
    ForecastBundle, build_ml_forecasts, build_periodic_forecasts,
    dispatch_recourse, dispatch_tracking, generate_conditioned_scenarios, solve_day_ahead,
    validate_day_ahead,
)


OPT_DIR = ROOT / "tmp" / "问题二_两日滚动实时纠偏"
OLD_BASELINE_COST = 14_640_006.236455053


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_from_old(path: Path, name: str) -> dict:
    raw = _json(path)
    return {
        "方案": name,
        "预测模型": "周期加权",
        "SOC逻辑": "6000日备惩罚" if "无日备" not in path.name else "无终端价值",
        "执行逻辑": "日前充放电固定",
        "计划购电量_kWh": raw["计划购电量_kWh"],
        "实际紧急购电量_kWh": raw["实际紧急购电量_kWh"],
        "实际富余电量_kWh": raw["实际富余电量_kWh"],
        "充电量_kWh": raw["充电量_kWh"],
        "放电量_kWh": raw["放电量_kWh"],
        "正常购电费_元": raw["正常购电费_元"],
        "紧急购电费_元": raw["紧急购电费_元"],
        "实际总费用_元": raw["实际总费用_元"],
        "平均日末储电量_kWh": raw["平均日末储电量_kWh"],
        "日末SOC标准差_kWh": 0.0,
        "约束通过": bool(raw["约束校验"]["all_days_feasible"]),
    }


def run_variant(data, forecasts: ForecastBundle, start_day: int = 1, end_day: int | None = None,
                terminal_under_value: float = 0.0, recourse_mode: str = "tracking") -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    end_day = len(data.dates) - 1 if end_day is None else end_day
    initial_soc = SOC_INITIAL
    details: list[dict] = []
    daily: list[dict] = []
    emergency_intervals: list[dict] = []
    checks: list[dict] = []
    scenario_audit: list[dict] = []
    started_all = time.perf_counter()

    for day in range(start_day, end_day + 1):
        date = data.dates[day]
        load_s, pv_s, probabilities, sampled, calibration = generate_conditioned_scenarios(data, day, forecasts)
        plan = solve_day_ahead(
            data.price, load_s, pv_s, probabilities, initial_soc,
            forecasts.next_load[day], forecasts.next_pv[day], terminal_soc=SOC_INITIAL,
        )
        plan_check = validate_day_ahead(plan, initial_soc, load_s, pv_s)
        if not plan_check["feasible"]:
            raise RuntimeError(f"{date.date()} 日前计划不可行: {plan_check}")

        actual_charge = np.zeros(N); actual_discharge = np.zeros(N)
        actual_soc = np.zeros(N); actual_emergency = np.zeros(N); actual_surplus = np.zeros(N)
        step_seconds = np.zeros(N); terminal_shortfall = np.zeros(N)
        observed_errors: list[float] = []
        soc = initial_soc
        for period in range(N):
            error = ((data.load_kwh[day, period] - data.pv_kwh[day, period])
                     - (forecasts.load[day, period] - forecasts.pv[day, period]))
            observed_errors.append(float(error))
            if recourse_mode == "tracking":
                step = dispatch_tracking(
                    period, data.load_kwh[day, period], data.pv_kwh[day, period], soc,
                    plan.purchase, forecasts.load[day], forecasts.pv[day],
                    plan.plan_charge, plan.plan_discharge,
                )
            else:
                step = dispatch_recourse(
                    period, data.load_kwh[day, period], data.pv_kwh[day, period], soc,
                    plan.purchase, forecasts.load[day], forecasts.pv[day], data.price,
                    plan.terminal_soc, np.asarray(observed_errors), terminal_under_value,
                )
            actual_charge[period] = step.charge; actual_discharge[period] = step.discharge
            actual_soc[period] = step.next_soc; actual_emergency[period] = step.emergency
            actual_surplus[period] = step.surplus; step_seconds[period] = step.solve_seconds
            terminal_shortfall[period] = step.terminal_shortfall
            soc = step.next_soc

        balance = (plan.purchase + actual_emergency + data.pv_kwh[day] - actual_surplus
                   + actual_discharge - data.load_kwh[day] - actual_charge)
        previous = np.r_[initial_soc, actual_soc[:-1]]
        soc_residual = actual_soc - previous - 0.9 * actual_charge + actual_discharge / 0.9
        simultaneous = (actual_charge > CHECK_TOL) & (actual_discharge > CHECK_TOL)
        actual_check = {
            "max_abs_actual_balance_residual_kwh": float(np.max(np.abs(balance))),
            "max_abs_actual_soc_recursion_kwh": float(np.max(np.abs(soc_residual))),
            "min_actual_soc_kwh": float(actual_soc.min()),
            "max_actual_soc_kwh": float(actual_soc.max()),
            "max_actual_charge_kwh": float(actual_charge.max()),
            "max_actual_discharge_kwh": float(actual_discharge.max()),
            "simultaneous_actual_charge_discharge_count": int(simultaneous.sum()),
        }
        actual_check["feasible"] = bool(
            actual_check["max_abs_actual_balance_residual_kwh"] <= CHECK_TOL
            and actual_check["max_abs_actual_soc_recursion_kwh"] <= CHECK_TOL
            and actual_check["min_actual_soc_kwh"] >= 1200 - CHECK_TOL
            and actual_check["max_actual_soc_kwh"] <= 10800 + CHECK_TOL
            and actual_check["max_actual_charge_kwh"] <= 5000/6 + CHECK_TOL
            and actual_check["max_actual_discharge_kwh"] <= 5000/6 + CHECK_TOL
            and actual_check["simultaneous_actual_charge_discharge_count"] == 0
        )
        if not actual_check["feasible"]:
            raise RuntimeError(f"{date.date()} 实时执行不可行: {actual_check}")

        plan_cost = float(data.price @ plan.purchase)
        emergency_cost = float((5.0 * data.price) @ actual_emergency)
        total_cost = plan_cost + emergency_cost
        curtailment = np.minimum(data.pv_kwh[day], actual_surplus)
        emergency_intervals.extend(merge_emergency_intervals(date, actual_emergency, data.interval_labels))

        for t in range(N):
            details.append({
                "日期": date, "时段编号": t + 1, "时间段": data.interval_labels[t],
                "电价(元/kWh)": data.price[t],
                "实际负荷(kWh)": data.load_kwh[day,t], "预测负荷(kWh)": forecasts.load[day,t],
                "实际光伏(kWh)": data.pv_kwh[day,t], "预测光伏(kWh)": forecasts.pv[day,t],
                "计划购电量(kWh)": plan.purchase[t],
                "基准充电量(kWh)": plan.plan_charge[t], "基准放电量(kWh)": plan.plan_discharge[t],
                "实际充电量(kWh)": actual_charge[t], "实际放电量(kWh)": actual_discharge[t],
                "充电纠偏量(kWh)": actual_charge[t]-plan.plan_charge[t],
                "放电纠偏量(kWh)": actual_discharge[t]-plan.plan_discharge[t],
                "时段初储电量(kWh)": initial_soc if t == 0 else actual_soc[t-1],
                "时段末储电量(kWh)": actual_soc[t], "基准时段末SOC(kWh)": plan.plan_soc[t],
                "实际紧急购电量(kWh)": actual_emergency[t], "实际富余电量(kWh)": actual_surplus[t],
                "实际弃光量(kWh)": curtailment[t],
                "正常购电费(元)": data.price[t]*plan.purchase[t],
                "紧急购电费(元)": 5*data.price[t]*actual_emergency[t],
                "纠偏信息截止时段": t + 1,
                "紧急购电原因": "SOC或放电功率受限" if actual_emergency[t] > CHECK_TOL else "无",
                "纠偏LP耗时(s)": step_seconds[t],
            })
        daily.append({
            "日期": date, "日初储电量(kWh)": initial_soc, "计划日末SOC(kWh)": plan.terminal_soc,
            "日末储电量(kWh)": actual_soc[-1], "预测负荷量(kWh)": forecasts.load[day].sum(),
            "实际负荷量(kWh)": data.load_kwh[day].sum(), "预测光伏量(kWh)": forecasts.pv[day].sum(),
            "实际光伏量(kWh)": data.pv_kwh[day].sum(), "计划购电量(kWh)": plan.purchase.sum(),
            "实际紧急购电量(kWh)": actual_emergency.sum(), "实际富余电量(kWh)": actual_surplus.sum(),
            "实际弃光量(kWh)": curtailment.sum(), "充电量(kWh)": actual_charge.sum(),
            "放电量(kWh)": actual_discharge.sum(), "正常购电费(元)": plan_cost,
            "紧急购电费(元)": emergency_cost, "实际总费用(元)": total_cost,
            "日前目标值(元)": plan.objective_yuan, "预期紧急购电费(元)": plan.expected_emergency_cost_yuan,
            "次日成本到达项(元)": plan.next_day_cost_yuan, "场景0.8分位覆盖率": calibration["q80_coverage"],
            "日前求解耗时(s)": plan.solve_seconds, "纠偏求解耗时(s)": step_seconds.sum(),
            "MIP相对间隙": plan.mip_gap,
        })
        checks.append({"日期": date.strftime("%Y-%m-%d"), **plan_check, **actual_check})
        scenario_audit.append({"日期": date.strftime("%Y-%m-%d"), **calibration,
                               "残差来源日期": [data.dates[i].strftime("%Y-%m-%d") for i in sampled]})
        initial_soc = float(actual_soc[-1])
        if day % 14 == 0 or day == end_day:
            print(f"[{forecasts.name}] {date.date()} 已完成，当日总费用={total_cost:.2f}，日末SOC={initial_soc:.1f}", flush=True)

    detail = pd.DataFrame(details); day_frame = pd.DataFrame(daily)
    official_detail = detail[(detail["日期"] >= OFFICIAL_START) & (detail["日期"] <= OFFICIAL_END)]
    official_daily = day_frame[(day_frame["日期"] >= OFFICIAL_START) & (day_frame["日期"] <= OFFICIAL_END)]
    metrics = {
        "负荷": prediction_metrics(data.load_kwh[31:end_day+1], forecasts.load[31:end_day+1]),
        "光伏_全时段": prediction_metrics(data.pv_kwh[31:end_day+1], forecasts.pv[31:end_day+1]),
    }
    max_check = {
        "all_days_feasible": all(x["feasible"] for x in checks),
        "max_abs_scenario_balance_kwh": max(x["max_abs_scenario_balance_kwh"] for x in checks),
        "max_abs_plan_soc_recursion_kwh": max(x["max_abs_plan_soc_recursion_kwh"] for x in checks),
        "max_abs_actual_balance_residual_kwh": max(x["max_abs_actual_balance_residual_kwh"] for x in checks),
        "max_abs_actual_soc_recursion_kwh": max(x["max_abs_actual_soc_recursion_kwh"] for x in checks),
        "simultaneous_plan_charge_discharge_count": sum(x["simultaneous_plan_charge_discharge_count"] for x in checks),
        "simultaneous_actual_charge_discharge_count": sum(x["simultaneous_actual_charge_discharge_count"] for x in checks),
        "min_actual_soc_kwh": min(x["min_actual_soc_kwh"] for x in checks),
        "max_actual_soc_kwh": max(x["max_actual_soc_kwh"] for x in checks),
        "max_mip_gap": max((x["mip_gap"] or 0.0) for x in checks),
    }
    summary = {
        "方案": f"两日SOC价值+实时纠偏+{forecasts.name}", "预测模型": forecasts.name,
        "SOC逻辑": "48小时滚动，仅次日末6000",
        "执行逻辑": "每10分钟轨迹跟踪误差纠偏" if recourse_mode == "tracking" else "每10分钟剩余时域纠偏",
        "实时层日末SOC不足影子价格_元每kWh": terminal_under_value,
        "正式区间": f"{OFFICIAL_START.date()}至{OFFICIAL_END.date()}", "正式天数": int(len(official_daily)),
        "预测指标": metrics,
        "计划购电量_kWh": float(official_daily["计划购电量(kWh)"].sum()),
        "实际紧急购电量_kWh": float(official_daily["实际紧急购电量(kWh)"].sum()),
        "实际富余电量_kWh": float(official_daily["实际富余电量(kWh)"].sum()),
        "实际弃光量_kWh": float(official_daily["实际弃光量(kWh)"].sum()),
        "充电量_kWh": float(official_daily["充电量(kWh)"].sum()), "放电量_kWh": float(official_daily["放电量(kWh)"].sum()),
        "正常购电费_元": float(official_daily["正常购电费(元)"].sum()),
        "紧急购电费_元": float(official_daily["紧急购电费(元)"].sum()),
        "实际总费用_元": float(official_daily["实际总费用(元)"].sum()),
        "平均日末储电量_kWh": float(official_daily["日末储电量(kWh)"].mean()),
        "日末SOC标准差_kWh": float(official_daily["日末储电量(kWh)"].std(ddof=0)),
        "发生紧急购电的时段数": int((official_detail["实际紧急购电量(kWh)"] > CHECK_TOL).sum()),
        "2月1日初始储电量_kWh": float(official_daily.iloc[0]["日初储电量(kWh)"]),
        "12月31日末储电量_kWh": float(official_daily.iloc[-1]["日末储电量(kWh)"]),
        "平均场景0.8分位覆盖率": float(official_daily["场景0.8分位覆盖率"].mean()),
        "约束校验": max_check, "约束通过": bool(max_check["all_days_feasible"]),
        "全年运行耗时_s": time.perf_counter() - started_all,
    }
    return detail, day_frame, summary, emergency_intervals


def write_variant(name: str, detail: pd.DataFrame, daily: pd.DataFrame, summary: dict, audit: list[dict]) -> None:
    folder = OPT_DIR / name; folder.mkdir(parents=True, exist_ok=True)
    detail.to_csv(folder / "逐时段结果.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(folder / "逐日汇总.csv", index=False, encoding="utf-8-sig")
    write_json(folder / "汇总与校验.json", summary)
    write_json(folder / "紧急购电区间.json", audit)


def draw_figures(best_detail: pd.DataFrame, best_daily: pd.DataFrame, ablation: list[dict]) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "font.size": 9})
    labels=[x["方案"] for x in ablation]; normal=np.array([x["正常购电费_元"] for x in ablation])/1e4
    emergency=np.array([x["紧急购电费_元"] for x in ablation])/1e4
    fig,ax=plt.subplots(figsize=(10.2,4.8),constrained_layout=True)
    x=np.arange(len(labels)); ax.bar(x,normal,color="#4C78A8",label="正常购电费")
    ax.bar(x,emergency,bottom=normal,color="#E45756",label="紧急购电费")
    ax.set_xticks(x,[f"方案{i+1}" for i in range(len(labels))]); ax.set_ylabel("费用/万元")
    for i,v in enumerate(normal+emergency): ax.text(i,v+5,f"{v:.1f}",ha="center")
    ax.legend(frameon=False,ncol=2); ax.grid(axis="y",ls="--",alpha=.4); ax.spines[["top","right"]].set_visible(False)
    p1=FIGURE_DIR/"问题二_四方案实际费用消融对比.pdf"; fig.savefig(p1,bbox_inches="tight"); plt.close(fig)

    official=best_daily[(best_daily["日期"]>=OFFICIAL_START)&(best_daily["日期"]<=OFFICIAL_END)]
    fig,axes=plt.subplots(2,1,figsize=(10.2,6.3),sharex=True,constrained_layout=True)
    axes[0].plot(official["日期"],official["日末储电量(kWh)"],color="#1F77B4",lw=1.1,label="实际日末SOC")
    axes[0].plot(official["日期"],official["计划日末SOC(kWh)"],color="#7F7F7F",lw=.8,alpha=.7,label="计划日末SOC")
    axes[0].axhline(6000,color="#444",ls="--",lw=.7,label="6000 kWh参考线")
    axes[0].set_ylabel("日末SOC/kWh"); axes[0].legend(frameon=False,ncol=3)
    axes[1].fill_between(official["日期"],0,official["实际紧急购电量(kWh)"],color="#E45756",alpha=.55)
    axes[1].set_ylabel("紧急购电量/(kWh/日)"); axes[1].set_xlabel("日期")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    for ax in axes: ax.grid(ls="--",alpha=.35); ax.spines[["top","right"]].set_visible(False)
    p2=FIGURE_DIR/"问题二_两日滚动SOC与实时纠偏.pdf"; fig.savefig(p2,bbox_inches="tight"); plt.close(fig)
    return [p1,p2]


def build_formal_bundle(detail: pd.DataFrame, daily: pd.DataFrame, summary: dict,
                        emergency_rows: list[dict], ablation: list[dict], figures: list[Path]) -> dict:
    official_detail=detail[(detail["日期"]>=OFFICIAL_START)&(detail["日期"]<=OFFICIAL_END)]
    official_daily=daily[(daily["日期"]>=OFFICIAL_START)&(daily["日期"]<=OFFICIAL_END)]
    purchase_rows=[]; storage_rows=[]
    for _,row in official_daily.iterrows():
        date=pd.Timestamp(row["日期"]); d=official_detail[official_detail["日期"]==date]
        purchase_rows.append({"date":date.strftime("%Y-%m-%d"),"purchase":d["计划购电量(kWh)"].tolist(),
                              "total_purchase":float(row["计划购电量(kWh)"]),"total_cost":float(row["实际总费用(元)"])})
        storage_rows.append({"date":date.strftime("%Y-%m-%d"),
            "charge_4h":[float(d["实际充电量(kWh)"].iloc[i:i+24].sum()) for i in range(0,N,24)],
            "discharge_4h":[float(d["实际放电量(kWh)"].iloc[i:i+24].sum()) for i in range(0,N,24)],
            "initial_soc":float(row["日初储电量(kWh)"]),"terminal_soc":float(row["日末储电量(kWh)"])})
    summary=dict(summary); summary["论文图"]=[str(p.relative_to(ROOT)) for p in figures]
    return {"purchaseRows":purchase_rows,"storageRows":storage_rows,
            "emergencyRows":[r for r in emergency_rows if OFFICIAL_START<=pd.Timestamp(r["日期"])<=OFFICIAL_END],
            "summary":summary,"ablation":ablation}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--variants",choices=["periodic","ml","all"],default="all")
    parser.add_argument("--start-day",type=int,default=1)
    parser.add_argument("--end-day",type=int,default=364)
    parser.add_argument("--terminal-under-value",type=float,default=0.0)
    parser.add_argument("--recourse-mode",choices=["tracking","mpc"],default="tracking")
    args=parser.parse_args()
    data=load_problem2_data(); OPT_DIR.mkdir(parents=True,exist_ok=True)
    periodic=build_periodic_forecasts(data)
    candidates=[periodic]
    if args.variants in ("ml","all"):
        candidates=[build_ml_forecasts(data,periodic)] if args.variants=="ml" else [periodic,build_ml_forecasts(data,periodic)]

    runs=[]
    for forecasts in candidates:
        detail,daily,summary,emergency=run_variant(data,forecasts,args.start_day,args.end_day,args.terminal_under_value,args.recourse_mode)
        write_variant(forecasts.name,detail,daily,summary,emergency)
        write_json(OPT_DIR/f"{forecasts.name}_预测审计.json",forecasts.audit)
        runs.append((forecasts,detail,daily,summary,emergency))

    old_reserve=_summary_from_old(INTERMEDIATE_DIR/"问题二_汇总与校验.json","方案1：日备惩罚+固定轨迹")
    old_no=_summary_from_old(INTERMEDIATE_DIR/"问题二_无日备惩罚_汇总与校验.json","方案2：无日备+固定轨迹")
    new_summaries=[]
    for i,(_,_,_,summary,_) in enumerate(runs,3):
        item=dict(summary); item["方案"]=f"方案{i}：{summary['方案']}"; new_summaries.append(item)
    ablation=[old_reserve,old_no,*new_summaries]
    best=min(runs,key=lambda x:x[3]["实际总费用_元"])
    figures=draw_figures(best[1],best[2],ablation)
    selected=best[3]["实际总费用_元"] < OLD_BASELINE_COST-CHECK_TOL
    decision={"候选方案":ablation,"最佳新方案":best[3]["方案"],"是否优于原正式方案":selected,
              "费用改善_元":OLD_BASELINE_COST-best[3]["实际总费用_元"]}
    write_json(OPT_DIR/"四方案消融与正式选择.json",decision)
    if selected and args.start_day==1 and args.end_day==364:
        bundle=build_formal_bundle(best[1],best[2],best[3],best[4],ablation,figures)
        write_json(INTERMEDIATE_DIR/"问题二_工作簿数据.json",bundle)
        best[1].to_csv(INTERMEDIATE_DIR/"问题二_两日滚动实时纠偏_逐时段结果.csv",index=False,encoding="utf-8-sig")
        best[2].to_csv(INTERMEDIATE_DIR/"问题二_两日滚动实时纠偏_逐日汇总.csv",index=False,encoding="utf-8-sig")
    print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
