"""问题一 LP/MILP 共用的数据、模型、校验与结果导出函数。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "predata" / "问题一_优化模型标准输入.xlsx"
TEMPLATE_FILE = ROOT / "附件" / "附件5" / "result1.xlsx"
OUTPUT_DIR = ROOT / "tmp" / "问题一_求解中间结果"

N = 144
DT_HOURS = 1 / 6
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_INITIAL = 6000.0
SOC_TERMINAL = 6000.0
POWER_MAX_KW = 5000.0
ENERGY_MAX = POWER_MAX_KW * DT_HOURS
ETA_C = 0.9
ETA_D = 0.9
CHECK_TOL = 1e-6


@dataclass(frozen=True)
class InputData:
    interval_id: np.ndarray
    raw_time: list[str]
    model_interval: list[str]
    template_interval: list[str]
    price: np.ndarray
    load_kw: np.ndarray
    pv_kw: np.ndarray
    load_kwh: np.ndarray
    pv_kwh: np.ndarray


@dataclass
class Solution:
    method: str
    status: int
    message: str
    objective_yuan: float
    purchase: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    curtailment: np.ndarray
    mode: np.ndarray | None
    solve_seconds: float
    solver_details: dict


def load_input() -> InputData:
    """按稳定的列位置读取预处理工作簿，规避工作表显示名称的编码差异。"""
    frame = pd.read_excel(INPUT_FILE, sheet_name=0, header=4)
    if frame.shape[0] != N or frame.shape[1] < 13:
        raise ValueError(f"标准输入应至少为 144 行、13 列，实际为 {frame.shape}")

    def numeric(column_index: int) -> np.ndarray:
        values = pd.to_numeric(frame.iloc[:, column_index], errors="raise").to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"第 {column_index + 1} 列存在非有限值")
        return values

    data = InputData(
        interval_id=numeric(0).astype(int),
        raw_time=frame.iloc[:, 1].astype(str).tolist(),
        model_interval=frame.iloc[:, 4].astype(str).tolist(),
        template_interval=frame.iloc[:, 5].astype(str).tolist(),
        price=numeric(7),
        load_kw=numeric(8),
        pv_kw=numeric(9),
        load_kwh=numeric(10),
        pv_kwh=numeric(11),
    )
    if not np.array_equal(data.interval_id, np.arange(1, N + 1)):
        raise ValueError("时段编号不是 1 至 144 的连续序列")
    if np.any(data.price < 0) or np.any(data.load_kwh <= 0) or np.any(data.pv_kwh < 0):
        raise ValueError("价格、负荷或光伏数据违反基本物理边界")
    return data


def _indices(with_binary: bool) -> dict[str, np.ndarray]:
    cursor = 0
    result: dict[str, np.ndarray] = {}
    for name in ["purchase", "charge", "discharge", "soc", "curtailment"]:
        result[name] = np.arange(cursor, cursor + N)
        cursor += N
    if with_binary:
        result["mode"] = np.arange(cursor, cursor + N)
    return result


def _equalities(data: InputData, idx: dict[str, np.ndarray], variable_count: int):
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row = 0

    # G - C + D - W = L - P
    for t in range(N):
        for column, value in [
            (idx["purchase"][t], 1.0),
            (idx["charge"][t], -1.0),
            (idx["discharge"][t], 1.0),
            (idx["curtailment"][t], -1.0),
        ]:
            rows.append(row); cols.append(int(column)); vals.append(value)
        rhs.append(float(data.load_kwh[t] - data.pv_kwh[t]))
        row += 1

    # B_t - B_(t-1) - eta_c C_t + D_t/eta_d = 0；t=0 时右端为 B_0。
    for t in range(N):
        terms = [
            (idx["soc"][t], 1.0),
            (idx["charge"][t], -ETA_C),
            (idx["discharge"][t], 1.0 / ETA_D),
        ]
        if t > 0:
            terms.append((idx["soc"][t - 1], -1.0))
        for column, value in terms:
            rows.append(row); cols.append(int(column)); vals.append(value)
        rhs.append(SOC_INITIAL if t == 0 else 0.0)
        row += 1

    # 日末储电量回到日初值。
    rows.append(row); cols.append(int(idx["soc"][-1])); vals.append(1.0)
    rhs.append(SOC_TERMINAL)
    matrix = coo_matrix((vals, (rows, cols)), shape=(row + 1, variable_count)).tocsr()
    return matrix, np.asarray(rhs)


def solve_lp(data: InputData) -> Solution:
    idx = _indices(with_binary=False)
    variable_count = 5 * N
    objective = np.zeros(variable_count)
    objective[idx["purchase"]] = data.price
    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    upper[idx["charge"]] = ENERGY_MAX
    upper[idx["discharge"]] = ENERGY_MAX
    lower[idx["soc"]] = SOC_MIN
    upper[idx["soc"]] = SOC_MAX
    upper[idx["curtailment"]] = data.pv_kwh
    a_eq, b_eq = _equalities(data, idx, variable_count)

    started = datetime.now()
    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=list(zip(lower, upper)),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    elapsed = (datetime.now() - started).total_seconds()
    if not result.success:
        raise RuntimeError(f"LP 求解失败：{result.status} {result.message}")
    x = result.x
    return Solution(
        method="LP",
        status=int(result.status),
        message=str(result.message),
        objective_yuan=float(result.fun),
        purchase=x[idx["purchase"]], charge=x[idx["charge"]],
        discharge=x[idx["discharge"]], soc=x[idx["soc"]],
        curtailment=x[idx["curtailment"]], mode=None,
        solve_seconds=elapsed,
        solver_details={"nit": int(result.nit), "crossover_nit": int(result.crossover_nit)},
    )


def solve_milp(data: InputData) -> Solution:
    idx = _indices(with_binary=True)
    variable_count = 6 * N
    objective = np.zeros(variable_count)
    objective[idx["purchase"]] = data.price
    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    upper[idx["charge"]] = ENERGY_MAX
    upper[idx["discharge"]] = ENERGY_MAX
    lower[idx["soc"]] = SOC_MIN
    upper[idx["soc"]] = SOC_MAX
    upper[idx["curtailment"]] = data.pv_kwh
    upper[idx["mode"]] = 1.0
    a_eq, b_eq = _equalities(data, idx, variable_count)

    # C_t <= Cmax z_t；D_t <= Dmax(1-z_t)。z=1 为充电许可，z=0 为放电许可。
    rows = np.repeat(np.arange(2 * N), 2)
    cols = np.empty(4 * N, dtype=int)
    vals = np.empty(4 * N)
    for t in range(N):
        cols[4*t:4*t+4] = [idx["charge"][t], idx["mode"][t], idx["discharge"][t], idx["mode"][t]]
        vals[4*t:4*t+4] = [1.0, -ENERGY_MAX, 1.0, ENERGY_MAX]
    a_ub = coo_matrix((vals, (rows, cols)), shape=(2 * N, variable_count)).tocsr()
    ub = np.r_[np.zeros(N), np.full(N, ENERGY_MAX)]
    # 上面 rows 的交错顺序对应 [charge row, discharge row]，因此重新构造右端。
    ub = np.tile([0.0, ENERGY_MAX], N)
    constraints = [
        LinearConstraint(a_eq, b_eq, b_eq),
        LinearConstraint(a_ub, np.full(2 * N, -np.inf), ub),
    ]
    integrality = np.zeros(variable_count, dtype=int)
    integrality[idx["mode"]] = 1

    started = datetime.now()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": 300.0, "mip_rel_gap": 0.0, "presolve": True},
    )
    elapsed = (datetime.now() - started).total_seconds()
    if not result.success:
        raise RuntimeError(f"MILP 求解失败：{result.status} {result.message}")
    x = result.x
    details = {}
    for name in ["mip_node_count", "mip_dual_bound", "mip_gap"]:
        if hasattr(result, name):
            value = getattr(result, name)
            details[name] = float(value) if value is not None else None
    return Solution(
        method="MILP", status=int(result.status), message=str(result.message),
        objective_yuan=float(result.fun), purchase=x[idx["purchase"]],
        charge=x[idx["charge"]], discharge=x[idx["discharge"]], soc=x[idx["soc"]],
        curtailment=x[idx["curtailment"]], mode=np.rint(x[idx["mode"]]).astype(int),
        solve_seconds=elapsed, solver_details=details,
    )


def check_solution(data: InputData, solution: Solution) -> dict:
    previous_soc = np.r_[SOC_INITIAL, solution.soc[:-1]]
    balance_residual = (
        solution.purchase + data.pv_kwh - solution.curtailment + solution.discharge
        - data.load_kwh - solution.charge
    )
    soc_residual = solution.soc - previous_soc - ETA_C * solution.charge + solution.discharge / ETA_D
    simultaneous = (solution.charge > CHECK_TOL) & (solution.discharge > CHECK_TOL)
    recalculated_cost = float(data.price @ solution.purchase)
    result = {
        "feasible": bool(
            np.max(np.abs(balance_residual)) <= CHECK_TOL
            and np.max(np.abs(soc_residual)) <= CHECK_TOL
            and np.min(solution.soc) >= SOC_MIN - CHECK_TOL
            and np.max(solution.soc) <= SOC_MAX + CHECK_TOL
            and np.max(solution.charge) <= ENERGY_MAX + CHECK_TOL
            and np.max(solution.discharge) <= ENERGY_MAX + CHECK_TOL
            and abs(solution.soc[-1] - SOC_TERMINAL) <= CHECK_TOL
        ),
        "max_abs_power_balance_residual_kwh": float(np.max(np.abs(balance_residual))),
        "max_abs_soc_recursion_residual_kwh": float(np.max(np.abs(soc_residual))),
        "min_soc_kwh": float(np.min(solution.soc)),
        "max_soc_kwh": float(np.max(solution.soc)),
        "terminal_soc_kwh": float(solution.soc[-1]),
        "max_charge_kwh_per_interval": float(np.max(solution.charge)),
        "max_discharge_kwh_per_interval": float(np.max(solution.discharge)),
        "simultaneous_charge_discharge_count": int(np.sum(simultaneous)),
        "simultaneous_interval_ids": data.interval_id[simultaneous].astype(int).tolist(),
        "max_min_charge_discharge_kwh": float(np.max(np.minimum(solution.charge, solution.discharge))),
        "objective_yuan": solution.objective_yuan,
        "recalculated_cost_yuan": recalculated_cost,
        "objective_recalculation_difference_yuan": float(solution.objective_yuan - recalculated_cost),
        "total_purchase_kwh": float(np.sum(solution.purchase)),
        "total_charge_kwh": float(np.sum(solution.charge)),
        "total_discharge_kwh": float(np.sum(solution.discharge)),
        "total_pv_curtailment_kwh": float(np.sum(solution.curtailment)),
        "solve_seconds": solution.solve_seconds,
        "solver_status": solution.status,
        "solver_message": solution.message,
        "solver_details": solution.solver_details,
    }
    return result


def detail_frame(data: InputData, solution: Solution) -> pd.DataFrame:
    previous_soc = np.r_[SOC_INITIAL, solution.soc[:-1]]
    balance_residual = solution.purchase + data.pv_kwh - solution.curtailment + solution.discharge - data.load_kwh - solution.charge
    frame = pd.DataFrame({
        "时段编号": data.interval_id,
        "原始时刻": data.raw_time,
        "模型时间段": data.model_interval,
        "提交模板时间段": data.template_interval,
        "电价(元/kWh)": data.price,
        "负荷功率(kW)": data.load_kw,
        "光伏功率(kW)": data.pv_kw,
        "负荷电量(kWh)": data.load_kwh,
        "光伏电量(kWh)": data.pv_kwh,
        "购电量(kWh)": solution.purchase,
        "充电量(kWh)": solution.charge,
        "放电量(kWh)": solution.discharge,
        "弃光量(kWh)": solution.curtailment,
        "时段初储电量(kWh)": previous_soc,
        "时段末储电量(kWh)": solution.soc,
        "时段购电费用(元)": data.price * solution.purchase,
        "电力平衡残差(kWh)": balance_residual,
        "同时充放电": (solution.charge > CHECK_TOL) & (solution.discharge > CHECK_TOL),
    })
    if solution.mode is not None:
        frame["充电许可二进制变量z"] = solution.mode
    return frame


def write_outputs(data: InputData, solution: Solution) -> tuple[Path, dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"问题一_{solution.method}_求解结果"
    frame = detail_frame(data, solution)
    checks = check_solution(data, solution)
    csv_path = OUTPUT_DIR / f"{stem}_逐时段.csv"
    json_path = OUTPUT_DIR / f"{stem}_校验摘要.json"
    bundle_path = OUTPUT_DIR / f"{stem}_工作簿数据.json"

    frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    four_hour = pd.DataFrame({
        "时间段": ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"],
        "充电量(kWh)": [solution.charge[i:i+24].sum() for i in range(0, N, 24)],
        "放电量(kWh)": [solution.discharge[i:i+24].sum() for i in range(0, N, 24)],
        "购电量(kWh)": [solution.purchase[i:i+24].sum() for i in range(0, N, 24)],
        "购电费用(元)": [(data.price[i:i+24] * solution.purchase[i:i+24]).sum() for i in range(0, N, 24)],
    })
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    bundle = {
        "method": solution.method,
        "checks": checks,
        "detailColumns": frame.columns.tolist(),
        "detailRows": frame.astype(object).where(pd.notna(frame), None).values.tolist(),
        "fourHourColumns": four_hour.columns.tolist(),
        "fourHourRows": four_hour.astype(object).where(pd.notna(four_hour), None).values.tolist(),
        "submission": {
            "purchase": solution.purchase.tolist(),
            "fourHourCharge": [float(solution.charge[i:i+24].sum()) for i in range(0, N, 24)],
            "fourHourDischarge": [float(solution.discharge[i:i+24].sum()) for i in range(0, N, 24)],
            "initialSoc": SOC_INITIAL,
            "terminalSoc": float(solution.soc[-1]),
        },
    }
    with bundle_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"method": solution.method, "data": str(bundle_path), "checks": checks}, ensure_ascii=False, indent=2))
    return bundle_path, checks
