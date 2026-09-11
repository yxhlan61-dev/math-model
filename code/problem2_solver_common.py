"""问题二：周期加权预测、联合残差场景与日备不足惩罚 MILP 公共实现。"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
ATTACHMENT1 = ROOT / "附件" / "附件1.xlsx"
ATTACHMENT2 = ROOT / "附件" / "附件2.xlsx"
OUTPUT_DIR = ROOT / "code" / "outputs" / "问题二"
INTERMEDIATE_DIR = ROOT / "tmp" / "问题二_求解中间结果"
FIGURE_DIR = ROOT / "figures" / "问题二"

N = 144
DT_HOURS = 1.0 / 6.0
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_INITIAL = 6000.0
POWER_MAX_KW = 5000.0
ENERGY_MAX = POWER_MAX_KW * DT_HOURS
ETA_C = 0.9
ETA_D = 0.9
RHO = 0.8
LOAD_CYCLES = 4
PV_CYCLES = 7
SCENARIO_COUNT = 100
RESERVE_TARGET = 6000.0
CHECK_TOL = 1e-6
OFFICIAL_START = pd.Timestamp("2025-02-01")
OFFICIAL_END = pd.Timestamp("2025-12-31")


@dataclass
class Problem2Data:
    dates: pd.DatetimeIndex
    interval_labels: list[str]
    price: np.ndarray
    load_kwh: np.ndarray
    pv_kwh: np.ndarray


@dataclass
class DailySolution:
    status: int
    message: str
    objective_yuan: float
    purchase: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    mode: np.ndarray
    emergency_scenarios: np.ndarray
    surplus_scenarios: np.ndarray
    expected_emergency_cost_yuan: float
    reserve_shortfall_kwh: float
    reserve_penalty_yuan: float
    solve_seconds: float
    solver_details: dict


def _interval_label(index_zero_based: int) -> str:
    start_total = (index_zero_based + 1) * 10
    end_total = (index_zero_based + 2) * 10

    def fmt(total: int) -> str:
        day = total // 1440
        minute = total % 1440
        hour, rem = divmod(minute, 60)
        suffix = "+1" if day else ""
        return f"{hour}:{rem:02d}{suffix}"

    return f"{fmt(start_total)}-{fmt(end_total)}"


def load_problem2_data() -> Problem2Data:
    price_frame = pd.read_excel(ATTACHMENT1)
    if price_frame.shape[0] != N:
        raise ValueError(f"附件1应有{N}个时段，实际为{price_frame.shape[0]}")
    price = pd.to_numeric(price_frame.iloc[:, 1], errors="raise").to_numpy(dtype=float)

    workbook = pd.ExcelFile(ATTACHMENT2)
    if len(workbook.sheet_names) != 2:
        raise ValueError(f"附件2应包含2个工作表，实际为{workbook.sheet_names}")
    load_frame = pd.read_excel(ATTACHMENT2, sheet_name=workbook.sheet_names[0])
    pv_frame = pd.read_excel(ATTACHMENT2, sheet_name=workbook.sheet_names[1])
    if load_frame.shape != (365, 145) or pv_frame.shape != (365, 145):
        raise ValueError(f"附件2形状异常：负荷{load_frame.shape}，光伏{pv_frame.shape}")

    load_dates = pd.to_datetime(load_frame.iloc[:, 0])
    pv_dates = pd.to_datetime(pv_frame.iloc[:, 0])
    if not load_dates.equals(pv_dates):
        raise ValueError("附件2负荷与光伏日期不一致")
    dates = pd.DatetimeIndex(load_dates)
    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    if not dates.equals(expected_dates):
        raise ValueError("附件2日期不是连续的2025全年日期")

    load_kw = load_frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    pv_kw = pv_frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(load_kw).all() or not np.isfinite(pv_kw).all():
        raise ValueError("附件2存在非有限数值")
    if load_kw.min() < -CHECK_TOL or pv_kw.min() < -CHECK_TOL:
        raise ValueError("附件2存在负的负荷或光伏功率")

    return Problem2Data(
        dates=dates,
        interval_labels=[_interval_label(i) for i in range(N)],
        price=price,
        load_kwh=load_kw * DT_HOURS,
        pv_kwh=pv_kw * DT_HOURS,
    )


def causal_forecasts(data: Problem2Data) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    days = len(data.dates)
    load_forecast = np.full((days, N), np.nan, dtype=float)
    pv_forecast = np.full((days, N), np.nan, dtype=float)
    audit: list[dict] = []

    for d in range(1, days):
        if d < 7:
            load_indices = np.arange(d - 1, max(-1, d - 8), -1, dtype=int)
            load_method = "首周日周期冷启动"
        else:
            k_count = min(LOAD_CYCLES, d // 7)
            load_indices = np.asarray([d - 7 * k for k in range(1, k_count + 1)], dtype=int)
            load_method = "负荷周周期加权平均"
        pv_indices = np.arange(d - 1, max(-1, d - PV_CYCLES - 1), -1, dtype=int)

        load_weights = RHO ** np.arange(len(load_indices), dtype=float)
        pv_weights = RHO ** np.arange(len(pv_indices), dtype=float)
        load_forecast[d] = np.average(data.load_kwh[load_indices], axis=0, weights=load_weights)
        pv_forecast[d] = np.average(data.pv_kwh[pv_indices], axis=0, weights=pv_weights)
        pv_forecast[d] = np.maximum(pv_forecast[d], 0.0)
        all_history_zero = np.all(data.pv_kwh[pv_indices] <= CHECK_TOL, axis=0)
        pv_forecast[d, all_history_zero] = 0.0

        audit.append({
            "预测日期": data.dates[d].strftime("%Y-%m-%d"),
            "预测原点": f"{data.dates[d].strftime('%Y-%m-%d')} 00:00:00",
            "训练数据最晚日期": data.dates[d - 1].strftime("%Y-%m-%d"),
            "负荷方法": load_method,
            "负荷历史日期": ",".join(data.dates[load_indices].strftime("%Y-%m-%d")),
            "光伏历史日期": ",".join(data.dates[pv_indices].strftime("%Y-%m-%d")),
            "衰减系数": RHO,
        })
    return load_forecast, pv_forecast, audit


def generate_scenarios(
    data: Problem2Data,
    day_index: int,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if day_index <= 1:
        return (
            load_forecast[day_index][None, :].copy(),
            pv_forecast[day_index][None, :].copy(),
            np.ones(1),
            [],
        )

    available = np.arange(1, day_index, dtype=int)
    if day_index >= 8:
        available = available[available >= 7]
    if len(available) == 0:
        return (
            load_forecast[day_index][None, :].copy(),
            pv_forecast[day_index][None, :].copy(),
            np.ones(1),
            [],
        )

    age = day_index - available
    same_weekday = (data.dates[available].weekday == data.dates[day_index].weekday).astype(float)
    weights = (0.98 ** age) * (1.0 + same_weekday)
    weights /= weights.sum()
    rng = np.random.default_rng(int(data.dates[day_index].strftime("%Y%m%d")))
    sampled = rng.choice(available, size=SCENARIO_COUNT, replace=True, p=weights)

    load_residual = data.load_kwh[sampled] - load_forecast[sampled]
    pv_residual = data.pv_kwh[sampled] - pv_forecast[sampled]
    load_scenarios = np.maximum(load_forecast[day_index][None, :] + load_residual, 0.0)
    historical_pv_max = float(data.pv_kwh[:day_index].max())
    pv_scenarios = np.clip(pv_forecast[day_index][None, :] + pv_residual, 0.0, historical_pv_max)
    night_mask = (
        (pv_forecast[day_index] <= CHECK_TOL)
        & np.all(data.pv_kwh[sampled] <= CHECK_TOL, axis=0)
    )
    pv_scenarios[:, night_mask] = 0.0
    probabilities = np.full(SCENARIO_COUNT, 1.0 / SCENARIO_COUNT)
    return load_scenarios, pv_scenarios, probabilities, sampled.tolist()


def _indices(scenario_count: int, include_reserve_shortfall: bool) -> tuple[dict[str, np.ndarray | int], int]:
    cursor = 0
    result: dict[str, np.ndarray | int] = {}
    for name in ["purchase", "charge", "discharge", "soc", "mode"]:
        result[name] = np.arange(cursor, cursor + N, dtype=int)
        cursor += N
    result["emergency"] = np.arange(cursor, cursor + scenario_count * N, dtype=int).reshape(scenario_count, N)
    cursor += scenario_count * N
    result["surplus"] = np.arange(cursor, cursor + scenario_count * N, dtype=int).reshape(scenario_count, N)
    cursor += scenario_count * N
    if include_reserve_shortfall:
        result["reserve_shortfall"] = cursor
        cursor += 1
    return result, cursor


def solve_daily_milp(
    price: np.ndarray,
    load_scenarios: np.ndarray,
    pv_scenarios: np.ndarray,
    probabilities: np.ndarray,
    initial_soc: float,
    terminal_soc: float | None,
    reserve_target: float | None = None,
    reserve_penalty: float = 0.0,
    time_limit_seconds: float = 120.0,
) -> DailySolution:
    scenario_count = load_scenarios.shape[0]
    if load_scenarios.shape != (scenario_count, N) or pv_scenarios.shape != (scenario_count, N):
        raise ValueError("场景矩阵形状错误")
    if probabilities.shape != (scenario_count,) or abs(probabilities.sum() - 1.0) > 1e-12:
        raise ValueError("场景概率错误")
    price_scenarios = np.asarray(price, dtype=float)
    if price_scenarios.ndim == 1:
        if price_scenarios.shape != (N,):
            raise ValueError("电价向量形状错误")
        price_scenarios = np.broadcast_to(price_scenarios, (scenario_count, N))
    elif price_scenarios.shape != (scenario_count, N):
        raise ValueError("电价场景矩阵形状错误")
    if not np.isfinite(price_scenarios).all() or np.min(price_scenarios) < -CHECK_TOL:
        raise ValueError("电价场景包含负值或非有限值")

    include_reserve_shortfall = terminal_soc is None and reserve_target is not None and reserve_penalty > 0.0
    idx, variable_count = _indices(scenario_count, include_reserve_shortfall)
    purchase = idx["purchase"]
    charge = idx["charge"]
    discharge = idx["discharge"]
    soc = idx["soc"]
    mode = idx["mode"]
    emergency = idx["emergency"]
    surplus = idx["surplus"]
    reserve_shortfall = int(idx["reserve_shortfall"]) if include_reserve_shortfall else None

    objective = np.zeros(variable_count)
    objective[purchase] = probabilities @ price_scenarios
    for s in range(scenario_count):
        objective[emergency[s]] = probabilities[s] * 5.0 * price_scenarios[s]
        objective[surplus[s]] = 0.0
    if reserve_shortfall is not None:
        objective[reserve_shortfall] = reserve_penalty

    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    lower[soc] = SOC_MIN
    upper[soc] = SOC_MAX
    upper[charge] = ENERGY_MAX
    upper[discharge] = ENERGY_MAX
    upper[mode] = 1.0
    integrality = np.zeros(variable_count, dtype=int)
    integrality[mode] = 1

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    eq_rhs: list[float] = []
    row = 0

    for s in range(scenario_count):
        for t in range(N):
            for column, value in [
                (purchase[t], 1.0),
                (emergency[s, t], 1.0),
                (surplus[s, t], -1.0),
                (discharge[t], 1.0),
                (charge[t], -1.0),
            ]:
                rows.append(row); cols.append(int(column)); values.append(value)
            eq_rhs.append(float(load_scenarios[s, t] - pv_scenarios[s, t]))
            row += 1

    for t in range(N):
        terms = [
            (soc[t], 1.0),
            (charge[t], -ETA_C),
            (discharge[t], 1.0 / ETA_D),
        ]
        if t > 0:
            terms.append((soc[t - 1], -1.0))
        for column, value in terms:
            rows.append(row); cols.append(int(column)); values.append(value)
        eq_rhs.append(float(initial_soc if t == 0 else 0.0))
        row += 1

    if terminal_soc is not None:
        rows.append(row); cols.append(int(soc[-1])); values.append(1.0)
        eq_rhs.append(float(terminal_soc))
        row += 1

    a_eq = coo_matrix((values, (rows, cols)), shape=(row, variable_count)).tocsr()

    rows = []
    cols = []
    values = []
    ub: list[float] = []
    row = 0
    for t in range(N):
        rows.extend([row, row]); cols.extend([int(charge[t]), int(mode[t])]); values.extend([1.0, -ENERGY_MAX])
        ub.append(0.0); row += 1
        rows.extend([row, row]); cols.extend([int(discharge[t]), int(mode[t])]); values.extend([1.0, ENERGY_MAX])
        ub.append(ENERGY_MAX); row += 1

    if reserve_shortfall is not None:
        # reserve_target - B_end <= U，即 -B_end-U <= -reserve_target。
        rows.extend([row, row])
        cols.extend([int(soc[-1]), reserve_shortfall])
        values.extend([-1.0, -1.0])
        ub.append(-float(reserve_target)); row += 1

    a_ub = coo_matrix((values, (rows, cols)), shape=(row, variable_count)).tocsr()
    constraints = [
        LinearConstraint(a_eq, np.asarray(eq_rhs), np.asarray(eq_rhs)),
        LinearConstraint(a_ub, np.full(row, -np.inf), np.asarray(ub)),
    ]

    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 1e-6, "presolve": True},
    )
    elapsed = time.perf_counter() - started
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"MILP求解失败：status={result.status}, {result.message}")

    x = result.x
    emergency_values = x[emergency]
    scenario_costs = np.sum(emergency_values * (5.0 * price_scenarios), axis=1)
    reserve_shortfall_value = 0.0 if reserve_shortfall is None else float(x[reserve_shortfall])
    details = {}
    for name in ["mip_node_count", "mip_dual_bound", "mip_gap"]:
        value = getattr(result, name, None)
        details[name] = None if value is None else float(value)

    return DailySolution(
        status=int(result.status),
        message=str(result.message),
        objective_yuan=float(result.fun),
        purchase=x[purchase],
        charge=x[charge],
        discharge=x[discharge],
        soc=x[soc],
        mode=np.rint(x[mode]).astype(int),
        emergency_scenarios=emergency_values,
        surplus_scenarios=x[surplus],
        expected_emergency_cost_yuan=float(probabilities @ scenario_costs),
        reserve_shortfall_kwh=reserve_shortfall_value,
        reserve_penalty_yuan=reserve_penalty * reserve_shortfall_value,
        solve_seconds=elapsed,
        solver_details=details,
    )


def validate_daily_solution(
    solution: DailySolution,
    price: np.ndarray,
    load_scenarios: np.ndarray,
    pv_scenarios: np.ndarray,
    initial_soc: float,
    terminal_soc: float | None,
) -> dict:
    previous_soc = np.r_[initial_soc, solution.soc[:-1]]
    soc_residual = solution.soc - previous_soc - ETA_C * solution.charge + solution.discharge / ETA_D
    balance = (
        solution.purchase[None, :] + solution.emergency_scenarios + pv_scenarios
        - solution.surplus_scenarios + solution.discharge[None, :]
        - load_scenarios - solution.charge[None, :]
    )
    simultaneous = (solution.charge > CHECK_TOL) & (solution.discharge > CHECK_TOL)
    result = {
        "max_abs_scenario_balance_residual_kwh": float(np.max(np.abs(balance))),
        "max_abs_soc_recursion_residual_kwh": float(np.max(np.abs(soc_residual))),
        "min_soc_kwh": float(solution.soc.min()),
        "max_soc_kwh": float(solution.soc.max()),
        "max_charge_kwh": float(solution.charge.max()),
        "max_discharge_kwh": float(solution.discharge.max()),
        "simultaneous_charge_discharge_count": int(simultaneous.sum()),
        "terminal_soc_difference_kwh": None if terminal_soc is None else float(solution.soc[-1] - terminal_soc),
        "solver_status": solution.status,
        "solver_message": solution.message,
        "mip_gap": solution.solver_details.get("mip_gap"),
    }
    result["feasible"] = bool(
        result["max_abs_scenario_balance_residual_kwh"] <= CHECK_TOL
        and result["max_abs_soc_recursion_residual_kwh"] <= CHECK_TOL
        and result["min_soc_kwh"] >= SOC_MIN - CHECK_TOL
        and result["max_soc_kwh"] <= SOC_MAX + CHECK_TOL
        and result["max_charge_kwh"] <= ENERGY_MAX + CHECK_TOL
        and result["max_discharge_kwh"] <= ENERGY_MAX + CHECK_TOL
        and result["simultaneous_charge_discharge_count"] == 0
        and (terminal_soc is None or abs(result["terminal_soc_difference_kwh"]) <= CHECK_TOL)
    )
    return result


def merge_emergency_intervals(date: pd.Timestamp, values: np.ndarray, labels: list[str]) -> list[dict]:
    active = values > CHECK_TOL
    rows: list[dict] = []
    start: int | None = None
    for t in range(N + 1):
        is_active = bool(active[t]) if t < N else False
        if is_active and start is None:
            start = t
        elif not is_active and start is not None:
            end = t - 1
            start_text = labels[start].split("-")[0]
            end_text = labels[end].split("-")[1]
            rows.append({
                "日期": date.strftime("%Y-%m-%d"),
                "紧急购电时间段": f"{start_text}-{end_text}",
                "紧急购电量(kWh)": float(values[start:t].sum()),
            })
            start = None
    return rows


def prediction_metrics(actual: np.ndarray, forecast: np.ndarray, pv: bool = False) -> dict:
    mask = np.isfinite(forecast)
    if pv:
        mask &= actual > CHECK_TOL
    errors = actual[mask] - forecast[mask]
    denominator = float(np.sum(np.abs(actual[mask])))
    return {
        "MAE(kWh)": float(np.mean(np.abs(errors))),
        "RMSE(kWh)": float(np.sqrt(np.mean(errors ** 2))),
        "WAPE": float(np.sum(np.abs(errors)) / denominator) if denominator > 0 else math.nan,
        "样本数": int(mask.sum()),
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
