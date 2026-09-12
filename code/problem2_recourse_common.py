"""问题二优化公共实现：48 小时日前滚动求解与 10 分钟因果储能纠偏。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix
from sklearn.ensemble import HistGradientBoostingRegressor

from problem2_solver_common import (
    CHECK_TOL,
    ENERGY_MAX,
    ETA_C,
    ETA_D,
    N,
    RHO,
    SCENARIO_COUNT,
    SOC_INITIAL,
    SOC_MAX,
    SOC_MIN,
    Problem2Data,
)


ML_REFIT_DAYS = 14
ML_MIN_HISTORY_DAYS = 28
SCENARIO_LOOKBACK_DAYS = 84
TERMINAL_UNDER_VALUE = 0.9 * 1.24995
SURPLUS_ABSORB_VALUE = 1e-5
THROUGHPUT_EPS = 1e-7


@dataclass
class ForecastBundle:
    name: str
    load: np.ndarray
    pv: np.ndarray
    next_load: np.ndarray
    next_pv: np.ndarray
    audit: list[dict]


@dataclass
class DayAheadSolution:
    objective_yuan: float
    purchase: np.ndarray
    plan_charge: np.ndarray
    plan_discharge: np.ndarray
    plan_soc: np.ndarray
    terminal_soc: float
    expected_emergency_cost_yuan: float
    next_day_cost_yuan: float
    scenario_emergency: np.ndarray
    scenario_surplus: np.ndarray
    solve_seconds: float
    mip_gap: float | None


@dataclass
class RecourseStep:
    charge: float
    discharge: float
    emergency: float
    surplus: float
    next_soc: float
    objective_yuan: float
    terminal_shortfall: float
    terminal_excess: float
    solve_seconds: float


def dispatch_tracking(
    period: int,
    actual_load: float,
    actual_pv: float,
    current_soc: float,
    locked_purchase: np.ndarray,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    plan_charge: np.ndarray,
    plan_discharge: np.ndarray,
) -> RecourseStep:
    """跟踪日前电价套利轨迹，仅用当期已观测净负荷误差修正储能动作。"""
    started = time.perf_counter()
    error = (actual_load - actual_pv) - (load_forecast[period] - pv_forecast[period])
    planned_net_charge = plan_charge[period] - plan_discharge[period]
    desired_net_charge = planned_net_charge - error
    charge_max = min(ENERGY_MAX, max((SOC_MAX - current_soc) / ETA_C, 0.0))
    discharge_max = min(ENERGY_MAX, max((current_soc - SOC_MIN) * ETA_D, 0.0))
    actual_net_charge = float(np.clip(desired_net_charge, -discharge_max, charge_max))
    charge = max(actual_net_charge, 0.0)
    discharge = max(-actual_net_charge, 0.0)
    next_soc = current_soc + ETA_C * charge - discharge / ETA_D
    residual = actual_load + charge - locked_purchase[period] - actual_pv - discharge
    emergency = max(residual, 0.0)
    surplus = max(-residual, 0.0)
    return RecourseStep(
        charge, discharge, emergency, surplus, float(next_soc),
        5.0 * emergency, 0.0, 0.0, time.perf_counter() - started,
    )


def _weighted_average(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    weights = RHO ** np.arange(len(indices), dtype=float)
    return np.average(values[indices], axis=0, weights=weights)


def periodic_forecasts_at_origin(data: Problem2Data, day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在 day 日 0:00 生成当日及次日预测；两者都只使用 day 之前数据。"""
    if day < 1:
        raise ValueError("至少需要一天历史数据")

    def one(target: int) -> tuple[np.ndarray, np.ndarray]:
        known_last = day - 1
        weekday = data.dates[0] + pd.Timedelta(days=target)
        candidates = [
            i for i in range(known_last, -1, -1)
            if data.dates[i].weekday() == weekday.weekday()
        ][:4]
        if not candidates:
            candidates = list(range(known_last, max(-1, known_last - 7), -1))
        pv_candidates = list(range(known_last, max(-1, known_last - 7), -1))
        load = _weighted_average(data.load_kwh, np.asarray(candidates, dtype=int))
        pv = np.maximum(_weighted_average(data.pv_kwh, np.asarray(pv_candidates, dtype=int)), 0.0)
        return load, pv

    current_load, current_pv = one(day)
    next_load, next_pv = one(day + 1)
    return current_load, current_pv, next_load, next_pv


def build_periodic_forecasts(data: Problem2Data) -> ForecastBundle:
    days = len(data.dates)
    load = np.full((days, N), np.nan)
    pv = np.full((days, N), np.nan)
    next_load = np.full((days, N), np.nan)
    next_pv = np.full((days, N), np.nan)
    audit: list[dict] = []
    for day in range(1, days):
        load[day], pv[day], next_load[day], next_pv[day] = periodic_forecasts_at_origin(data, day)
        audit.append({
            "预测日期": data.dates[day].strftime("%Y-%m-%d"),
            "模型": "周期加权平均",
            "信息截止日期": data.dates[day - 1].strftime("%Y-%m-%d"),
            "次日预测是否使用当日实际值": False,
        })
    return ForecastBundle("周期加权", load, pv, next_load, next_pv, audit)


def _calendar_features(date: pd.Timestamp, periods: np.ndarray) -> np.ndarray:
    tod = periods / N
    doy = (date.dayofyear - 1) / 365.0
    weekday = date.weekday()
    return np.column_stack([
        np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
        np.sin(4 * np.pi * tod), np.cos(4 * np.pi * tod),
        np.full(N, weekday / 6.0),
        np.full(N, np.sin(2 * np.pi * doy)),
        np.full(N, np.cos(2 * np.pi * doy)),
    ])


def _feature_day(values: np.ndarray, dates: pd.DatetimeIndex, target: int,
                 lag1_override: np.ndarray | None = None) -> np.ndarray:
    periods = np.arange(N, dtype=float)
    date = dates[0] + pd.Timedelta(days=target)
    lag1 = lag1_override if lag1_override is not None else values[target - 1]
    lag7 = values[target - 7] if target >= 7 else values[max(0, target - 1)]
    history = values[max(0, target - 28):target]
    last7 = history[-7:]
    return np.column_stack([
        _calendar_features(pd.Timestamp(date), periods),
        lag1,
        lag7,
        last7.mean(axis=0),
        last7.std(axis=0),
        history.mean(axis=0),
    ])


def _fit_hist_model(values: np.ndarray, dates: pd.DatetimeIndex, end_exclusive: int) -> HistGradientBoostingRegressor:
    start = 7
    x = np.vstack([_feature_day(values, dates, day) for day in range(start, end_exclusive)])
    y = np.concatenate([values[day] for day in range(start, end_exclusive)])
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.08, max_iter=80,
        max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=0.2,
        random_state=20250912,
    )
    return model.fit(x, y)


def build_ml_forecasts(data: Problem2Data, periodic: ForecastBundle) -> ForecastBundle:
    """严格扩展窗口 HistGradientBoosting；每 14 天重拟合，其余日期复用旧模型。"""
    days = len(data.dates)
    load = periodic.load.copy()
    pv = periodic.pv.copy()
    next_load = periodic.next_load.copy()
    next_pv = periodic.next_pv.copy()
    audit: list[dict] = []
    load_model = pv_model = None
    fitted_until = -1
    for day in range(1, days):
        use_ml = day >= ML_MIN_HISTORY_DAYS
        if use_ml and (load_model is None or day - fitted_until >= ML_REFIT_DAYS):
            load_model = _fit_hist_model(data.load_kwh, data.dates, day)
            pv_model = _fit_hist_model(data.pv_kwh, data.dates, day)
            fitted_until = day
        if use_ml:
            load[day] = np.maximum(load_model.predict(_feature_day(data.load_kwh, data.dates, day)), 0.0)
            pv[day] = np.maximum(pv_model.predict(_feature_day(data.pv_kwh, data.dates, day)), 0.0)
            next_load[day] = np.maximum(load_model.predict(
                _feature_day(data.load_kwh, data.dates, day + 1, lag1_override=load[day])
            ), 0.0)
            next_pv[day] = np.maximum(pv_model.predict(
                _feature_day(data.pv_kwh, data.dates, day + 1, lag1_override=pv[day])
            ), 0.0)
        audit.append({
            "预测日期": data.dates[day].strftime("%Y-%m-%d"),
            "模型": "HistGradientBoosting" if use_ml else "周期冷启动",
            "训练数据最晚日期": data.dates[day - 1].strftime("%Y-%m-%d"),
            "当前拟合截止索引": fitted_until if use_ml else None,
            "次日滞后1天特征使用当日预测": bool(use_ml),
            "次日预测是否使用当日实际值": False,
        })
    return ForecastBundle("因果梯度提升", load, pv, next_load, next_pv, audit)


def generate_conditioned_scenarios(
    data: Problem2Data, day: int, forecasts: ForecastBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict]:
    available = np.arange(max(1, day - SCENARIO_LOOKBACK_DAYS), day, dtype=int)
    if len(available) == 0:
        return (forecasts.load[day][None, :], forecasts.pv[day][None, :],
                np.ones(1), [], {"q80_coverage": np.nan, "pool_days": 0})
    age = day - available
    same_weekday = (data.dates[available].weekday == data.dates[day].weekday).astype(float)
    same_month = (data.dates[available].month == data.dates[day].month).astype(float)
    weights = (0.985 ** age) * (1.0 + same_weekday) * (1.0 + 0.35 * same_month)
    weights /= weights.sum()
    rng = np.random.default_rng(int(data.dates[day].strftime("%Y%m%d")) + (17 if forecasts.name.startswith("因果") else 0))
    sampled = rng.choice(available, size=SCENARIO_COUNT, replace=True, p=weights)
    load_res = data.load_kwh[sampled] - forecasts.load[sampled]
    pv_res = data.pv_kwh[sampled] - forecasts.pv[sampled]
    load_pool = data.load_kwh[available] - forecasts.load[available]
    pv_pool = data.pv_kwh[available] - forecasts.pv[available]
    net_pool = load_pool - pv_pool
    net_sample = load_res - pv_res
    # 对净负荷 0.8 分位做在线校准，不使用当日数据。
    shift = np.quantile(net_pool, 0.80, axis=0) - np.quantile(net_sample, 0.80, axis=0)
    load_scenarios = np.maximum(forecasts.load[day][None, :] + load_res + shift[None, :], 0.0)
    historical_pv_max = float(data.pv_kwh[:day].max())
    pv_scenarios = np.clip(forecasts.pv[day][None, :] + pv_res, 0.0, historical_pv_max)
    probabilities = np.full(SCENARIO_COUNT, 1.0 / SCENARIO_COUNT)
    q80 = np.quantile(load_scenarios - pv_scenarios, 0.80, axis=0)
    actual_net = data.load_kwh[day] - data.pv_kwh[day]
    audit = {
        "pool_days": int(len(available)),
        "q80_coverage": float(np.mean(actual_net <= q80)),
        "mean_q80_shift_kwh": float(np.mean(shift)),
    }
    return load_scenarios, pv_scenarios, probabilities, sampled.tolist(), audit


def _day_ahead_indices(scenarios: int) -> tuple[dict[str, np.ndarray], int]:
    cursor = 0
    idx: dict[str, np.ndarray] = {}
    for name in ["g1", "c1", "d1", "b1", "z1", "g2", "c2", "d2", "b2", "z2"]:
        idx[name] = np.arange(cursor, cursor + N, dtype=int); cursor += N
    for name in ["h1", "w1"]:
        idx[name] = np.arange(cursor, cursor + scenarios * N, dtype=int).reshape(scenarios, N)
        cursor += scenarios * N
    idx["w2"] = np.arange(cursor, cursor + N, dtype=int); cursor += N
    return idx, cursor


def solve_day_ahead(
    price: np.ndarray,
    load_scenarios: np.ndarray,
    pv_scenarios: np.ndarray,
    probabilities: np.ndarray,
    initial_soc: float,
    next_load_forecast: np.ndarray,
    next_pv_forecast: np.ndarray,
    terminal_soc: float = SOC_INITIAL,
    time_limit_seconds: float = 120.0,
) -> DayAheadSolution:
    """48 小时 MILP：当日场景调度 + 次日确定性成本到达项。"""
    S = load_scenarios.shape[0]
    idx, nvar = _day_ahead_indices(S)
    c = np.zeros(nvar)
    c[idx["g1"]] = price
    c[idx["g2"]] = price
    for s in range(S):
        c[idx["h1"][s]] = probabilities[s] * 5.0 * price
    c[idx["c1"]] = c[idx["d1"]] = THROUGHPUT_EPS
    c[idx["c2"]] = c[idx["d2"]] = THROUGHPUT_EPS

    lower = np.zeros(nvar); upper = np.full(nvar, np.inf)
    for name in ["b1", "b2"]:
        lower[idx[name]] = SOC_MIN; upper[idx[name]] = SOC_MAX
    for name in ["c1", "d1", "c2", "d2"]:
        upper[idx[name]] = ENERGY_MAX
    for name in ["z1", "z2"]:
        upper[idx[name]] = 1.0
    integrality = np.zeros(nvar, dtype=int)
    integrality[idx["z1"]] = 1; integrality[idx["z2"]] = 1

    er: list[int] = []; ec: list[int] = []; ev: list[float] = []; rhs: list[float] = []; row = 0
    def add_eq(terms, value):
        nonlocal row
        for col, coefficient in terms:
            er.append(row); ec.append(int(col)); ev.append(float(coefficient))
        rhs.append(float(value)); row += 1

    for s in range(S):
        for t in range(N):
            add_eq([(idx["g1"][t], 1), (idx["h1"][s,t], 1), (idx["w1"][s,t], -1),
                    (idx["d1"][t], 1), (idx["c1"][t], -1)],
                   load_scenarios[s,t] - pv_scenarios[s,t])
    for t in range(N):
        terms = [(idx["b1"][t], 1), (idx["c1"][t], -ETA_C), (idx["d1"][t], 1/ETA_D)]
        if t: terms.append((idx["b1"][t-1], -1))
        add_eq(terms, initial_soc if t == 0 else 0)
    for t in range(N):
        add_eq([(idx["g2"][t], 1), (idx["w2"][t], -1), (idx["d2"][t], 1), (idx["c2"][t], -1)],
               next_load_forecast[t] - next_pv_forecast[t])
        terms = [(idx["b2"][t], 1), (idx["c2"][t], -ETA_C), (idx["d2"][t], 1/ETA_D)]
        terms.append((idx["b1"][-1], -1) if t == 0 else (idx["b2"][t-1], -1))
        add_eq(terms, 0)
    add_eq([(idx["b2"][-1], 1)], terminal_soc)
    aeq = coo_matrix((ev, (er, ec)), shape=(row, nvar)).tocsr()

    ur: list[int] = []; uc: list[int] = []; uv: list[float] = []; ub: list[float] = []; row = 0
    def add_ub(terms, value):
        nonlocal row
        for col, coefficient in terms:
            ur.append(row); uc.append(int(col)); uv.append(float(coefficient))
        ub.append(float(value)); row += 1
    for prefix in ["1", "2"]:
        for t in range(N):
            add_ub([(idx[f"c{prefix}"][t], 1), (idx[f"z{prefix}"][t], -ENERGY_MAX)], 0)
            add_ub([(idx[f"d{prefix}"][t], 1), (idx[f"z{prefix}"][t], ENERGY_MAX)], ENERGY_MAX)
    aub = coo_matrix((uv, (ur, uc)), shape=(row, nvar)).tocsr()
    started = time.perf_counter()
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper), constraints=[
        LinearConstraint(aeq, np.asarray(rhs), np.asarray(rhs)),
        LinearConstraint(aub, np.full(row, -np.inf), np.asarray(ub)),
    ], options={"time_limit": time_limit_seconds, "mip_rel_gap": 1e-6, "presolve": True})
    elapsed = time.perf_counter() - started
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"48小时日前MILP失败: {result.status}, {result.message}")
    x = result.x
    expected_emergency = float(sum(
        probabilities[s] * np.dot(5.0 * price, x[idx["h1"][s]]) for s in range(S)
    ))
    return DayAheadSolution(
        float(result.fun), x[idx["g1"]], x[idx["c1"]], x[idx["d1"]], x[idx["b1"]],
        float(x[idx["b1"][-1]]), expected_emergency, float(np.dot(price, x[idx["g2"]])),
        x[idx["h1"]], x[idx["w1"]], elapsed,
        None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
    )


def dispatch_recourse(
    period: int,
    actual_load: float,
    actual_pv: float,
    current_soc: float,
    locked_purchase: np.ndarray,
    load_forecast: np.ndarray,
    pv_forecast: np.ndarray,
    price: np.ndarray,
    planned_terminal_soc: float,
    observed_net_errors: np.ndarray,
    terminal_under_value: float = TERMINAL_UNDER_VALUE,
) -> RecourseStep:
    """因果剩余时域 LP；只执行第一个 10 分钟动作。"""
    T = N - period
    # 已发生误差的指数衰减均值，只修正未来净负荷。
    if observed_net_errors.size:
        weights = 0.85 ** np.arange(observed_net_errors.size - 1, -1, -1)
        bias = float(np.dot(weights, observed_net_errors) / weights.sum())
    else:
        bias = 0.0
    net = load_forecast[period:] - pv_forecast[period:] - locked_purchase[period:]
    decay = np.exp(-np.arange(T) / 24.0)
    net = net + bias * decay
    net[0] = actual_load - actual_pv - locked_purchase[period]

    # c,d,b,h,w, terminal_under, terminal_over
    cursor = 0; idx = {}
    for name in ["c", "d", "b", "h", "w"]:
        idx[name] = np.arange(cursor, cursor + T); cursor += T
    under = cursor; over = cursor + 1; nvar = cursor + 2
    objective = np.zeros(nvar)
    objective[idx["c"]] = THROUGHPUT_EPS; objective[idx["d"]] = THROUGHPUT_EPS
    objective[idx["h"]] = 5.0 * price[period:]
    objective[idx["w"]] = SURPLUS_ABSORB_VALUE
    objective[under] = terminal_under_value
    objective[over] = 0.0
    bounds = [(0, ENERGY_MAX)] * T + [(0, ENERGY_MAX)] * T + [(SOC_MIN, SOC_MAX)] * T \
        + [(0, None)] * T + [(0, None)] * T + [(0, None), (0, None)]

    er=[]; ec=[]; ev=[]; rhs=[]; row=0
    for t in range(T):
        for col, val in [(idx["c"][t], -1), (idx["d"][t], 1), (idx["h"][t], 1), (idx["w"][t], -1)]:
            er.append(row); ec.append(int(col)); ev.append(val)
        rhs.append(float(net[t])); row += 1
    for t in range(T):
        terms=[(idx["b"][t],1),(idx["c"][t],-ETA_C),(idx["d"][t],1/ETA_D)]
        if t: terms.append((idx["b"][t-1],-1))
        for col,val in terms:
            er.append(row); ec.append(int(col)); ev.append(val)
        rhs.append(float(current_soc if t == 0 else 0)); row += 1
    for col,val in [(idx["b"][-1],1),(under,1),(over,-1)]:
        er.append(row); ec.append(int(col)); ev.append(val)
    rhs.append(float(planned_terminal_soc)); row += 1
    aeq=coo_matrix((ev,(er,ec)),shape=(row,nvar)).tocsr()
    started=time.perf_counter()
    result=linprog(objective,A_eq=aeq,b_eq=np.asarray(rhs),bounds=bounds,method="highs")
    elapsed=time.perf_counter()-started
    if not result.success or result.x is None:
        raise RuntimeError(f"实时纠偏LP失败 period={period}: {result.message}")
    x=result.x
    charge=float(x[idx["c"][0]]); discharge=float(x[idx["d"][0]])
    # HiGHS 退化解如出现极小同时充放电，按净能量无损折算。
    if charge > CHECK_TOL and discharge > CHECK_TOL:
        delta_soc = ETA_C * charge - discharge / ETA_D
        if delta_soc >= 0:
            charge = delta_soc / ETA_C; discharge = 0.0
        else:
            charge = 0.0; discharge = -delta_soc * ETA_D
    next_soc=float(current_soc + ETA_C*charge - discharge/ETA_D)
    residual=float(actual_load + charge - locked_purchase[period] - actual_pv - discharge)
    emergency=max(residual,0.0); surplus=max(-residual,0.0)
    return RecourseStep(charge,discharge,emergency,surplus,next_soc,float(result.fun),
                        float(x[under]),float(x[over]),elapsed)


def validate_day_ahead(solution: DayAheadSolution, initial_soc: float,
                       load_scenarios: np.ndarray | None = None,
                       pv_scenarios: np.ndarray | None = None) -> dict:
    previous=np.r_[initial_soc,solution.plan_soc[:-1]]
    recursion=solution.plan_soc-previous-ETA_C*solution.plan_charge+solution.plan_discharge/ETA_D
    simultaneous=(solution.plan_charge>CHECK_TOL)&(solution.plan_discharge>CHECK_TOL)
    scenario_balance = 0.0
    if load_scenarios is not None and pv_scenarios is not None:
        balance = (solution.purchase[None, :] + solution.scenario_emergency
                   - solution.scenario_surplus + solution.plan_discharge[None, :]
                   - solution.plan_charge[None, :] - load_scenarios + pv_scenarios)
        scenario_balance = float(np.max(np.abs(balance)))
    return {
        "max_abs_scenario_balance_kwh": scenario_balance,
        "max_abs_plan_soc_recursion_kwh": float(np.max(np.abs(recursion))),
        "min_plan_soc_kwh": float(solution.plan_soc.min()),
        "max_plan_soc_kwh": float(solution.plan_soc.max()),
        "simultaneous_plan_charge_discharge_count": int(simultaneous.sum()),
        "mip_gap": solution.mip_gap,
        "feasible": bool(scenario_balance <= CHECK_TOL
                         and np.max(np.abs(recursion)) <= CHECK_TOL and not simultaneous.any()
                         and solution.plan_soc.min() >= SOC_MIN-CHECK_TOL
                         and solution.plan_soc.max() <= SOC_MAX+CHECK_TOL),
    }
