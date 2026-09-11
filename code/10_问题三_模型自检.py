"""问题三数据映射、费用分段和滚动 MILP 的快速自检。"""

from __future__ import annotations

import numpy as np

from problem3_solver_common import (
    ENERGY_MAX,
    ISSUE_HOURS,
    RESERVE_PENALTY,
    generate_horizon_scenarios,
    load_forecast_only,
    load_problem3_data,
    solve_horizon_milp,
    validate_horizon,
)


def main() -> None:
    data = load_problem3_data()
    assert data.pv_hourly_forecast_kw.shape == (365, 4, 24)
    assert data.pv_interval_forecast_kwh.shape == (365, 4, 144)
    for issue_index, issue_hour in enumerate(ISSUE_HOURS):
        start = issue_hour * 6
        if start:
            assert np.isnan(data.pv_interval_forecast_kwh[:, issue_index, :start]).all()
        assert np.isfinite(data.pv_interval_forecast_kwh[:, issue_index, start:]).all()
        # 每个发布后的第一个整点必须精确落在第1小时预报上。
        first_hour_endpoint = (issue_hour + 1) * 6 - 1
        np.testing.assert_allclose(
            data.pv_interval_forecast_kwh[:, issue_index, first_hour_endpoint] * 6,
            data.pv_hourly_forecast_kw[:, issue_index, 0], atol=1e-10,
        )

    load_forecast, audit = load_forecast_only(data)
    assert len(audit) == 364
    scenarios = generate_horizon_scenarios(data, 31, 6, 36, load_forecast, scenario_count=10)
    load_s, pv_s, prob, sampled = scenarios
    assert load_s.shape == pv_s.shape
    assert load_s.shape[1] == 108
    assert 1 <= load_s.shape[0] <= 10
    assert abs(prob.sum() - 1.0) < 1e-12
    assert all(index < 31 for index in sampled)

    # 用短时域验证上调、下调与物理约束。
    price = np.array([0.5, 1.0, 0.8, 0.6])
    original = np.array([10.0, 10.0, 10.0, 10.0])
    load = np.array([[8.0, 14.0, 10.0, 9.0]])
    pv = np.zeros_like(load)
    solution = solve_horizon_milp(
        price, load, pv, np.ones(1), initial_soc=6000.0, terminal_soc=6000.0,
        original_plan=original, reserve_penalty=RESERVE_PENALTY,
    )
    checks = validate_horizon(solution, load, pv, 6000.0, original)
    assert checks["max_abs_scenario_balance_residual_kwh"] <= 1e-6
    assert checks["max_abs_soc_recursion_residual_kwh"] <= 1e-6
    assert checks["max_abs_adjustment_residual_kwh"] <= 1e-6
    assert checks["simultaneous_charge_discharge_count"] == 0
    assert checks["simultaneous_up_down_count"] == 0
    assert checks["max_charge_kwh"] <= ENERGY_MAX + 1e-6

    # 三段净结算公式的直接数值检验。
    lam = 2.0
    for adjusted, expected in [(6.0, 16.0), (10.0, 20.0), (14.0, 32.0)]:
        up = max(adjusted - 10.0, 0.0)
        down = max(10.0 - adjusted, 0.0)
        actual = lam * 10.0 + 1.5 * lam * up - 0.5 * lam * down
        assert abs(actual - expected) < 1e-12
    print("问题三模型自检通过")


if __name__ == "__main__":
    main()
