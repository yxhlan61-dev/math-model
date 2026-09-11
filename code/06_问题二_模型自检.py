"""问题二关键公式与小规模 MILP 自检。"""

from dataclasses import replace

import numpy as np

from problem2_solver_common import (
    CHECK_TOL,
    ENERGY_MAX,
    N,
    RHO,
    causal_forecasts,
    load_problem2_data,
    solve_daily_milp,
    validate_daily_solution,
)


def main() -> None:
    data = load_problem2_data()
    load_forecast, pv_forecast, audit = causal_forecasts(data)

    d = 35
    expected_load = (
        data.load_kwh[d - 7]
        + RHO * data.load_kwh[d - 14]
        + RHO ** 2 * data.load_kwh[d - 21]
        + RHO ** 3 * data.load_kwh[d - 28]
    ) / (1 + RHO + RHO ** 2 + RHO ** 3)
    expected_pv = sum(RHO ** k * data.pv_kwh[d - 1 - k] for k in range(7)) / sum(RHO ** k for k in range(7))
    assert np.max(np.abs(load_forecast[d] - expected_load)) <= 1e-12
    assert np.max(np.abs(pv_forecast[d] - expected_pv)) <= 1e-12
    assert pd_timestamp(audit[d - 1]["训练数据最晚日期"]) < data.dates[d]

    # 修改预测日及未来实际数据，预测日0:00的结果必须完全不变。
    modified_load = data.load_kwh.copy()
    modified_pv = data.pv_kwh.copy()
    modified_load[d:] += 1_000_000.0
    modified_pv[d:] += 1_000_000.0
    modified_data = replace(data, load_kwh=modified_load, pv_kwh=modified_pv)
    modified_load_forecast, modified_pv_forecast, _ = causal_forecasts(modified_data)
    assert np.array_equal(load_forecast[d], modified_load_forecast[d])
    assert np.array_equal(pv_forecast[d], modified_pv_forecast[d])

    load_scenarios = np.vstack([load_forecast[d], load_forecast[d] * 1.03])
    pv_scenarios = np.vstack([pv_forecast[d], pv_forecast[d] * 0.95])
    probabilities = np.array([0.5, 0.5])
    solution = solve_daily_milp(data.price, load_scenarios, pv_scenarios, probabilities, 6000.0, 6000.0)
    check = validate_daily_solution(solution, data.price, load_scenarios, pv_scenarios, 6000.0, 6000.0)
    assert check["feasible"], check
    assert check["simultaneous_charge_discharge_count"] == 0
    assert check["max_charge_kwh"] <= ENERGY_MAX + CHECK_TOL
    assert check["max_discharge_kwh"] <= ENERGY_MAX + CHECK_TOL

    reserve_solution = solve_daily_milp(
        data.price, load_scenarios, pv_scenarios, probabilities, 1200.0, None,
        reserve_target=6000.0, reserve_penalty=1.124955,
    )
    expected_shortfall = max(6000.0 - reserve_solution.soc[-1], 0.0)
    assert abs(reserve_solution.reserve_shortfall_kwh - expected_shortfall) <= CHECK_TOL
    assert abs(
        reserve_solution.reserve_penalty_yuan
        - 1.124955 * reserve_solution.reserve_shortfall_kwh
    ) <= CHECK_TOL
    print("问题二预测公式、信息边界和小规模MILP自检通过。")


def pd_timestamp(value):
    import pandas as pd
    return pd.Timestamp(value)


if __name__ == "__main__":
    main()
