"""问题三：官方光伏预报、联合残差场景与滚动调整 MILP 公共实现。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from problem2_solver_common import (
    CHECK_TOL,
    ENERGY_MAX,
    ETA_C,
    ETA_D,
    N,
    RESERVE_TARGET,
    SOC_INITIAL,
    SOC_MAX,
    SOC_MIN,
    Problem2Data,
    causal_forecasts,
    load_problem2_data,
)


ROOT = Path(__file__).resolve().parents[1]
ATTACHMENT3 = ROOT / "附件" / "附件3.xlsx"
OUTPUT_DIR = ROOT / "code" / "outputs" / "问题三"
INTERMEDIATE_DIR = ROOT / "tmp" / "问题三_求解中间结果"
FIGURE_DIR = ROOT / "figures" / "问题三"
ISSUE_HOURS = (0, 6, 12, 18)
SCENARIO_COUNT = 100
OFFICIAL_START = pd.Timestamp("2025-02-01")
OFFICIAL_END = pd.Timestamp("2025-12-31")
RESERVE_PENALTY = 1.124955


@dataclass
class Problem3Data:
    base: Problem2Data
    pv_hourly_forecast_kw: np.ndarray  # [day, issue, lead=1..24]
    pv_interval_forecast_kwh: np.ndarray  # [day, issue, same-day interval], past is NaN


@dataclass
class HorizonSolution:
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
    upward: np.ndarray
    downward: np.ndarray
    reserve_shortfall_kwh: float
    reserve_penalty_yuan: float
    terminal_excess_kwh: float
    terminal_excess_penalty_yuan: float
    expected_emergency_cost_yuan: float
    solve_seconds: float
    mip_gap: float | None


def load_problem3_data() -> Problem3Data:
    base = load_problem2_data()
    raw = pd.read_excel(ATTACHMENT3)
    expected_columns = ["日期", "预报时刻"] + [f"预报{k}小时" for k in range(1, 25)]
    if list(raw.columns) != expected_columns:
        raise ValueError(f"附件3字段异常：{list(raw.columns)}")
    if raw.shape != (365 * 4, 26):
        raise ValueError(f"附件3形状应为(1460,26)，实际为{raw.shape}")

    raw["日期"] = pd.to_datetime(raw["日期"].ffill())
    issue_map = {f"{hour}:00": i for i, hour in enumerate(ISSUE_HOURS)}
    if set(raw["预报时刻"].astype(str)) != set(issue_map):
        raise ValueError("附件3预报时刻不是每天0:00、6:00、12:00、18:00")
    if raw.duplicated(["日期", "预报时刻"]).any():
        raise ValueError("附件3存在重复的日期与预报时刻")

    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    hourly = np.full((365, 4, 24), np.nan, dtype=float)
    date_index = {date: i for i, date in enumerate(expected_dates)}
    for _, row in raw.iterrows():
        date = pd.Timestamp(row["日期"])
        if date not in date_index:
            raise ValueError(f"附件3日期越界：{date}")
        issue_text = str(row["预报时刻"])
        values = pd.to_numeric(row.iloc[2:], errors="raise").to_numpy(dtype=float)
        hourly[date_index[date], issue_map[issue_text], :] = values
    if not np.isfinite(hourly).all() or hourly.min() < -CHECK_TOL:
        raise ValueError("附件3存在缺失、非有限或负的光伏预报")

    interval = np.full((365, 4, N), np.nan, dtype=float)
    endpoint_hours = (np.arange(N, dtype=float) + 1.0) / 6.0
    for d in range(365):
        for issue_index, issue_hour in enumerate(ISSUE_HOURS):
            if issue_hour == 0:
                anchor_kw = 0.0 if d == 0 else float(base.pv_kwh[d - 1, -1] * 6.0)
            else:
                anchor_kw = float(base.pv_kwh[d, issue_hour * 6 - 1] * 6.0)
            knot_hours = issue_hour + np.arange(25, dtype=float)
            knot_values = np.r_[anchor_kw, hourly[d, issue_index]]
            future_mask = endpoint_hours > issue_hour + 1e-12
            interval[d, issue_index, future_mask] = np.interp(
                endpoint_hours[future_mask], knot_hours, knot_values
            ) / 6.0
    return Problem3Data(base=base, pv_hourly_forecast_kw=hourly, pv_interval_forecast_kwh=interval)


def load_forecast_only(data: Problem3Data) -> tuple[np.ndarray, list[dict]]:
    load_forecast, _unused_pv, audit = causal_forecasts(data.base)
    return load_forecast, audit


def generate_horizon_scenarios(
    data: Problem3Data,
    day_index: int,
    issue_hour: int,
    start_index: int,
    load_forecast: np.ndarray,
    scenario_count: int = SCENARIO_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    issue_index = ISSUE_HOURS.index(issue_hour)
    current_load = load_forecast[day_index, start_index:]
    current_pv = data.pv_interval_forecast_kwh[day_index, issue_index, start_index:]
    if not np.isfinite(current_load).all() or not np.isfinite(current_pv).all():
        raise ValueError(f"{data.base.dates[day_index].date()} {issue_hour}:00预测区间含缺失值")

    available = np.arange(1, day_index, dtype=int)
    if day_index >= 8:
        available = available[available >= 7]
    if len(available) == 0:
        return current_load[None, :], current_pv[None, :], np.ones(1), []

    age = day_index - available
    same_weekday = (data.base.dates[available].weekday == data.base.dates[day_index].weekday).astype(float)
    weights = (0.98 ** age) * (1.0 + same_weekday)
    weights /= weights.sum()
    seed = int(data.base.dates[day_index].strftime("%Y%m%d")) * 10 + issue_index
    rng = np.random.default_rng(seed)
    sampled_draws = rng.choice(available, size=scenario_count, replace=True, p=weights)
    # 有放回的100次抽样可能包含重复历史日。将完全相同的场景合并并按频数赋权，
    # 与保留100份重复变量的目标函数和可行域严格等价，但可显著缩小MILP。
    sampled, counts = np.unique(sampled_draws, return_counts=True)

    load_residual = data.base.load_kwh[sampled, start_index:] - load_forecast[sampled, start_index:]
    historical_pv_forecast = data.pv_interval_forecast_kwh[sampled, issue_index, start_index:]
    if not np.isfinite(historical_pv_forecast).all():
        raise ValueError("历史官方光伏预报残差区间不完整")
    pv_residual = data.base.pv_kwh[sampled, start_index:] - historical_pv_forecast
    load_scenarios = np.maximum(current_load[None, :] + load_residual, 0.0)

    past_observed = data.base.pv_kwh[:day_index].ravel()
    if start_index > 0:
        past_observed = np.r_[past_observed, data.base.pv_kwh[day_index, :start_index]]
    historical_pv_max = float(np.max(past_observed)) if len(past_observed) else float(np.max(current_pv))
    pv_scenarios = np.clip(current_pv[None, :] + pv_residual, 0.0, historical_pv_max)
    night_mask = (current_pv <= CHECK_TOL) & np.all(data.base.pv_kwh[sampled, start_index:] <= CHECK_TOL, axis=0)
    pv_scenarios[:, night_mask] = 0.0
    probabilities = counts.astype(float) / float(scenario_count)
    return load_scenarios, pv_scenarios, probabilities, sampled.tolist()


def _variable_indices(T: int, S: int, adjusted: bool, include_reserve: bool,
                      include_terminal_excess: bool) -> tuple[dict, int]:
    cursor = 0
    idx: dict[str, np.ndarray | int] = {}
    for name in ["purchase", "charge", "discharge", "soc", "mode"]:
        idx[name] = np.arange(cursor, cursor + T, dtype=int)
        cursor += T
    idx["emergency"] = np.arange(cursor, cursor + S * T, dtype=int).reshape(S, T)
    cursor += S * T
    if adjusted:
        idx["upward"] = np.arange(cursor, cursor + T, dtype=int); cursor += T
        idx["downward"] = np.arange(cursor, cursor + T, dtype=int); cursor += T
    if include_reserve:
        idx["reserve"] = cursor; cursor += 1
    if include_terminal_excess:
        idx["terminal_excess"] = cursor; cursor += 1
    return idx, cursor


def solve_horizon_milp(
    price: np.ndarray,
    load_scenarios: np.ndarray,
    pv_scenarios: np.ndarray,
    probabilities: np.ndarray,
    initial_soc: float,
    terminal_soc: float | None,
    original_plan: np.ndarray | None = None,
    reserve_target: float | None = RESERVE_TARGET,
    reserve_penalty: float = RESERVE_PENALTY,
    terminal_upper_target: float | None = None,
    terminal_upper_penalty: float = 0.0,
    time_limit_seconds: float = 120.0,
) -> HorizonSolution:
    S, T = load_scenarios.shape
    if pv_scenarios.shape != (S, T) or probabilities.shape != (S,):
        raise ValueError("滚动MILP输入维度不一致")
    if abs(float(probabilities.sum()) - 1.0) > 1e-12:
        raise ValueError("场景概率和不为1")
    price_scenarios = np.asarray(price, dtype=float)
    if price_scenarios.ndim == 1:
        if price_scenarios.shape != (T,):
            raise ValueError("滚动电价向量长度错误")
        price_scenarios = np.broadcast_to(price_scenarios, (S, T))
    elif price_scenarios.shape != (S, T):
        raise ValueError("滚动电价场景矩阵形状错误")
    if not np.isfinite(price_scenarios).all() or np.min(price_scenarios) < -CHECK_TOL:
        raise ValueError("滚动电价场景包含负值或非有限值")
    expected_price = probabilities @ price_scenarios
    adjusted = original_plan is not None
    if adjusted and original_plan.shape != (T,):
        raise ValueError("原计划长度与剩余时域不一致")
    include_reserve = terminal_soc is None and reserve_target is not None and reserve_penalty > 0
    include_terminal_excess = terminal_soc is None and terminal_upper_target is not None and terminal_upper_penalty > 0
    idx, nvar = _variable_indices(T, S, adjusted, include_reserve, include_terminal_excess)
    purchase = idx["purchase"]; charge = idx["charge"]; discharge = idx["discharge"]
    soc = idx["soc"]; mode = idx["mode"]; emergency = idx["emergency"]

    objective = np.zeros(nvar)
    if adjusted:
        upward = idx["upward"]; downward = idx["downward"]
        objective[upward] = 1.5 * expected_price
        objective[downward] = -0.5 * expected_price
    else:
        upward = downward = None
        objective[purchase] = expected_price
    for s in range(S):
        objective[emergency[s]] = probabilities[s] * 5.0 * price_scenarios[s]
    if include_reserve:
        objective[int(idx["reserve"])] = reserve_penalty
    if include_terminal_excess:
        objective[int(idx["terminal_excess"])] = terminal_upper_penalty

    lower = np.zeros(nvar)
    upper = np.full(nvar, np.inf)
    lower[soc] = SOC_MIN; upper[soc] = SOC_MAX
    upper[charge] = ENERGY_MAX; upper[discharge] = ENERGY_MAX; upper[mode] = 1.0
    integrality = np.zeros(nvar, dtype=int); integrality[mode] = 1

    eq_rows: list[int] = []; eq_cols: list[int] = []; eq_values: list[float] = []; eq_rhs: list[float] = []
    row = 0
    for t in range(T):
        terms = [(soc[t], 1.0), (charge[t], -ETA_C), (discharge[t], 1.0 / ETA_D)]
        if t > 0:
            terms.append((soc[t - 1], -1.0))
        for column, value in terms:
            eq_rows.append(row); eq_cols.append(int(column)); eq_values.append(value)
        eq_rhs.append(float(initial_soc if t == 0 else 0.0)); row += 1
    if adjusted:
        for t in range(T):
            for column, value in [(purchase[t], 1.0), (upward[t], -1.0), (downward[t], 1.0)]:
                eq_rows.append(row); eq_cols.append(int(column)); eq_values.append(value)
            eq_rhs.append(float(original_plan[t])); row += 1
    if terminal_soc is not None:
        eq_rows.append(row); eq_cols.append(int(soc[-1])); eq_values.append(1.0)
        eq_rhs.append(float(terminal_soc)); row += 1
    a_eq = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(row, nvar)).tocsr()

    ub_rows: list[int] = []; ub_cols: list[int] = []; ub_values: list[float] = []; ub_rhs: list[float] = []
    row_ub = 0
    # H >= L-P+C-D-A。富余量零成本且不参与其他约束，可在求解后由正部恢复，
    # 无需为每个场景时段显式增加W变量与等式。
    for s in range(S):
        for t in range(T):
            for column, value in [(purchase[t], -1.0), (emergency[s, t], -1.0),
                                  (discharge[t], -1.0), (charge[t], 1.0)]:
                ub_rows.append(row_ub); ub_cols.append(int(column)); ub_values.append(value)
            ub_rhs.append(float(-(load_scenarios[s, t] - pv_scenarios[s, t]))); row_ub += 1
    for t in range(T):
        ub_rows.extend([row_ub, row_ub]); ub_cols.extend([int(charge[t]), int(mode[t])]); ub_values.extend([1.0, -ENERGY_MAX])
        ub_rhs.append(0.0); row_ub += 1
        ub_rows.extend([row_ub, row_ub]); ub_cols.extend([int(discharge[t]), int(mode[t])]); ub_values.extend([1.0, ENERGY_MAX])
        ub_rhs.append(ENERGY_MAX); row_ub += 1
    if include_reserve:
        ub_rows.extend([row_ub, row_ub]); ub_cols.extend([int(soc[-1]), int(idx["reserve"])])
        ub_values.extend([-1.0, -1.0]); ub_rhs.append(-float(reserve_target)); row_ub += 1
    if include_terminal_excess:
        ub_rows.extend([row_ub, row_ub]); ub_cols.extend([int(soc[-1]), int(idx["terminal_excess"])])
        ub_values.extend([1.0, -1.0]); ub_rhs.append(float(terminal_upper_target)); row_ub += 1
    a_ub = coo_matrix((ub_values, (ub_rows, ub_cols)), shape=(row_ub, nvar)).tocsr()
    constraints = [
        LinearConstraint(a_eq, np.asarray(eq_rhs), np.asarray(eq_rhs)),
        LinearConstraint(a_ub, np.full(row_ub, -np.inf), np.asarray(ub_rhs)),
    ]

    started = time.perf_counter()
    result = milp(c=objective, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints,
                  options={"time_limit": time_limit_seconds, "mip_rel_gap": 1e-6, "presolve": True})
    elapsed = time.perf_counter() - started
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"滚动MILP求解失败：status={result.status}, {result.message}")
    x = result.x
    upward_values = np.zeros(T) if not adjusted else x[upward]
    downward_values = np.zeros(T) if not adjusted else x[downward]
    reserve_value = 0.0 if not include_reserve else float(x[int(idx["reserve"])])
    terminal_excess_value = 0.0 if not include_terminal_excess else float(x[int(idx["terminal_excess"])])
    scenario_emergency = x[emergency]
    purchase_values = x[purchase]
    charge_values = x[charge]
    discharge_values = x[discharge]
    surplus_values = np.maximum(
        purchase_values[None, :] + pv_scenarios + discharge_values[None, :]
        - load_scenarios - charge_values[None, :], 0.0
    )
    return HorizonSolution(
        status=int(result.status), message=str(result.message), objective_yuan=float(result.fun),
        purchase=purchase_values, charge=charge_values, discharge=discharge_values, soc=x[soc],
        mode=np.rint(x[mode]).astype(int), emergency_scenarios=scenario_emergency,
        surplus_scenarios=surplus_values, upward=upward_values, downward=downward_values,
        reserve_shortfall_kwh=reserve_value, reserve_penalty_yuan=reserve_penalty * reserve_value,
        terminal_excess_kwh=terminal_excess_value,
        terminal_excess_penalty_yuan=terminal_upper_penalty * terminal_excess_value,
        expected_emergency_cost_yuan=float(probabilities @ np.sum(
            scenario_emergency * (5.0 * price_scenarios), axis=1
        )),
        solve_seconds=elapsed, mip_gap=None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
    )


def validate_horizon(solution: HorizonSolution, load_scenarios: np.ndarray, pv_scenarios: np.ndarray,
                     initial_soc: float, original_plan: np.ndarray | None = None) -> dict:
    previous = np.r_[initial_soc, solution.soc[:-1]]
    soc_residual = solution.soc - previous - ETA_C * solution.charge + solution.discharge / ETA_D
    balance = (solution.purchase[None, :] + solution.emergency_scenarios + pv_scenarios
               - solution.surplus_scenarios + solution.discharge[None, :]
               - load_scenarios - solution.charge[None, :])
    adjustment_residual = 0.0
    if original_plan is not None:
        adjustment_residual = float(np.max(np.abs(
            solution.purchase - original_plan - solution.upward + solution.downward
        )))
    simultaneous = int(np.sum((solution.charge > CHECK_TOL) & (solution.discharge > CHECK_TOL)))
    return {
        "max_abs_scenario_balance_residual_kwh": float(np.max(np.abs(balance))),
        "max_abs_soc_recursion_residual_kwh": float(np.max(np.abs(soc_residual))),
        "max_abs_adjustment_residual_kwh": adjustment_residual,
        "min_soc_kwh": float(solution.soc.min()), "max_soc_kwh": float(solution.soc.max()),
        "max_charge_kwh": float(solution.charge.max()), "max_discharge_kwh": float(solution.discharge.max()),
        "simultaneous_charge_discharge_count": simultaneous,
        "simultaneous_up_down_count": int(np.sum((solution.upward > CHECK_TOL) & (solution.downward > CHECK_TOL))),
        "mip_gap": solution.mip_gap,
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
