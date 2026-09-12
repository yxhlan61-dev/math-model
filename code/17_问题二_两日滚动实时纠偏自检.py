"""问题二新策略的信息边界、物理约束和纠偏方向自检。"""

from dataclasses import replace

import numpy as np

from problem2_solver_common import CHECK_TOL, N, SOC_INITIAL, load_problem2_data
from problem2_recourse_common import (
    build_periodic_forecasts, dispatch_recourse, dispatch_tracking, generate_conditioned_scenarios,
    solve_day_ahead, validate_day_ahead,
)


def main() -> None:
    data = load_problem2_data()
    forecast = build_periodic_forecasts(data)
    day = 31

    # 改动当日及以后实际数据，0:00 形成的预测和场景必须不变。
    changed = replace(data, load_kwh=data.load_kwh.copy(), pv_kwh=data.pv_kwh.copy())
    changed.load_kwh[day:] += 9999.0
    changed.pv_kwh[day:] *= 0.0
    forecast_changed = build_periodic_forecasts(changed)
    assert np.allclose(forecast.load[day], forecast_changed.load[day])
    assert np.allclose(forecast.next_load[day], forecast_changed.next_load[day])
    original_s = generate_conditioned_scenarios(data, day, forecast)
    changed_s = generate_conditioned_scenarios(changed, day, forecast_changed)
    assert np.allclose(original_s[0], changed_s[0])
    assert np.allclose(original_s[1], changed_s[1])

    load_s, pv_s, probabilities, _, _ = original_s
    solution = solve_day_ahead(
        data.price, load_s, pv_s, probabilities, SOC_INITIAL,
        forecast.next_load[day], forecast.next_pv[day],
    )
    check = validate_day_ahead(solution, SOC_INITIAL, load_s, pv_s)
    assert check["feasible"], check
    assert abs(solution.terminal_soc - SOC_INITIAL) > CHECK_TOL, "当日日末SOC仍被锁定在6000"

    purchase = np.zeros(N)
    zeros = np.zeros(N)
    price = np.ones(N)
    # 正净负荷且电池可放电时，应先放电而不是紧急购电。
    positive = dispatch_recourse(0, 100.0, 0.0, 6000.0, purchase, zeros, zeros,
                                 price, 6000.0, np.array([100.0]))
    assert positive.discharge > 99.0 and positive.emergency <= CHECK_TOL, positive
    # 负净负荷且电池可充电时，应先充电吸收富余。
    negative = dispatch_recourse(0, 0.0, 100.0, 6000.0, purchase, zeros, zeros,
                                 price, 6000.0, np.array([-100.0]))
    assert negative.charge > 99.0 and negative.surplus <= CHECK_TOL, negative
    # SOC 下限不允许继续放电，缺口必须成为紧急购电。
    limited = dispatch_recourse(143, 100.0, 0.0, 1200.0, purchase, zeros, zeros,
                                price, 1200.0, np.array([100.0]))
    assert limited.discharge <= CHECK_TOL and limited.emergency > 99.0, limited
    tracking = dispatch_tracking(0, 120.0, 0.0, 6000.0, purchase, zeros, zeros,
                                 zeros, zeros)
    assert tracking.discharge > 119.0 and tracking.emergency <= CHECK_TOL, tracking
    print("问题二两日滚动与实时纠偏自检通过。")


if __name__ == "__main__":
    main()
