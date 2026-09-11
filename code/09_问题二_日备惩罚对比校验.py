"""从逐日结果独立复算有、无日备不足惩罚的费用与储能差异。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from problem2_solver_common import CHECK_TOL, INTERMEDIATE_DIR, OFFICIAL_END, OFFICIAL_START


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    official = frame[(frame["日期"] >= OFFICIAL_START) & (frame["日期"] <= OFFICIAL_END)]
    return {
        "计划购电量(kWh)": float(official["计划购电量(kWh)"].sum()),
        "实际紧急购电量(kWh)": float(official["实际紧急购电量(kWh)"].sum()),
        "充电量(kWh)": float(official["充电量(kWh)"].sum()),
        "放电量(kWh)": float(official["放电量(kWh)"].sum()),
        "正常购电费(元)": float(official["正常购电费(元)"].sum()),
        "紧急购电费(元)": float(official["紧急购电费(元)"].sum()),
        "实际购电总费用(元)": float(official["实际总费用(元)"].sum()),
        "平均日末储电量(kWh)": float(official["日末储电量(kWh)"].mean()),
        "日末储电量标准差(kWh)": float(official["日末储电量(kWh)"].std(ddof=0)),
        "日末低于6000kWh天数": int((official["日末储电量(kWh)"] < 6000 - CHECK_TOL).sum()),
        "12月31日末储电量(kWh)": float(official.iloc[-1]["日末储电量(kWh)"]),
    }


def check_continuity(frame: pd.DataFrame) -> float:
    previous_end = frame["日末储电量(kWh)"].to_numpy()[:-1]
    next_start = frame["日初储电量(kWh)"].to_numpy()[1:]
    return float(np.max(np.abs(previous_end - next_start)))


def main() -> None:
    with_reserve = pd.read_csv(INTERMEDIATE_DIR / "问题二_逐日汇总.csv", parse_dates=["日期"])
    without_reserve = pd.read_csv(
        INTERMEDIATE_DIR / "问题二_无日备惩罚_逐日汇总.csv", parse_dates=["日期"]
    )
    if not with_reserve["日期"].equals(without_reserve["日期"]):
        raise AssertionError("两套方案日期不一致")

    with_summary = summarize(with_reserve)
    without_summary = summarize(without_reserve)
    comparison = pd.DataFrame({
        "指标": list(with_summary),
        "无日备不足惩罚": [without_summary[key] for key in with_summary],
        "有日备不足惩罚": [with_summary[key] for key in with_summary],
        "差值(有-无)": [with_summary[key] - without_summary[key] for key in with_summary],
    })

    with_continuity = check_continuity(with_reserve)
    without_continuity = check_continuity(without_reserve)
    if with_continuity > CHECK_TOL or without_continuity > CHECK_TOL:
        raise AssertionError("跨日SOC连续性校验失败")
    if abs(with_summary["平均日末储电量(kWh)"] - 6000.0) > CHECK_TOL:
        raise AssertionError("有日备不足惩罚方案未达到日备目标")
    if abs(without_summary["平均日末储电量(kWh)"] - 1200.0) > CHECK_TOL:
        raise AssertionError("无日备不足惩罚基准不符合预期")

    output = INTERMEDIATE_DIR / "问题二_日备不足惩罚方案对比.csv"
    comparison.to_csv(output, index=False, encoding="utf-8-sig")
    print(comparison.to_string(index=False))
    print(f"跨日SOC连续性最大误差：无惩罚={without_continuity:.3e}，有惩罚={with_continuity:.3e}")
    print(output)


if __name__ == "__main__":
    main()
