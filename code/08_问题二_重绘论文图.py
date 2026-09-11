"""不重复求解 MILP，利用已保存逐时段结果重绘问题二论文图。"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

from problem2_solver_common import INTERMEDIATE_DIR, causal_forecasts, load_problem2_data


module_path = Path(__file__).with_name("05_问题二_周期预测与风险MILP.py")
spec = importlib.util.spec_from_file_location("problem2_main", module_path)
problem2_main = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(problem2_main)
draw_figures = problem2_main.draw_figures
draw_weekday_figure = problem2_main.draw_weekday_figure
draw_intraday_figure = problem2_main.draw_intraday_figure
draw_yearly_figure = problem2_main.draw_yearly_figure
draw_comparison_figure = problem2_main.draw_comparison_figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yearly-only", action="store_true", help="只重绘全年购电与日末储电量图")
    parser.add_argument("--optimization-only", action="store_true", help="只重绘受日备惩罚影响的优化结果图")
    parser.add_argument("--intraday-only", action="store_true", help="只重绘3月20日24小时图")
    parser.add_argument(
        "--weekday-comparison-only",
        action="store_true",
        help="只重绘公历星期曲线、日备惩罚对比图和3月20日24小时图",
    )
    args = parser.parse_args()
    detail = pd.read_csv(INTERMEDIATE_DIR / "问题二_逐时段结果.csv", parse_dates=["日期"])
    if args.intraday_only:
        print(draw_intraday_figure(detail))
    elif args.weekday_comparison_only:
        data = load_problem2_data()
        baseline_daily = pd.read_csv(
            INTERMEDIATE_DIR / "问题二_无日备惩罚_逐日汇总.csv", parse_dates=["日期"]
        )
        current_daily = pd.read_csv(
            INTERMEDIATE_DIR / "问题二_逐日汇总.csv", parse_dates=["日期"]
        )
        print(draw_weekday_figure(data))
        print(draw_comparison_figure(current_daily, baseline_daily))
        print(draw_intraday_figure(detail))
    elif args.optimization_only:
        baseline_daily = pd.read_csv(
            INTERMEDIATE_DIR / "问题二_无日备惩罚_逐日汇总.csv", parse_dates=["日期"]
        )
        current_daily = detail.groupby("日期", as_index=False).agg(
            **{
                "日末储电量(kWh)": ("时段末储电量(kWh)", "last"),
                "实际总费用(元)": ("正常购电费(元)", "sum"),
            }
        )
        emergency_cost = detail.groupby("日期")["紧急购电费(元)"].sum().to_numpy()
        current_daily["实际总费用(元)"] += emergency_cost
        print(draw_yearly_figure(detail))
        print(draw_comparison_figure(current_daily, baseline_daily))
    elif args.yearly_only:
        print(draw_yearly_figure(detail))
    else:
        data = load_problem2_data()
        load_forecast, pv_forecast, _ = causal_forecasts(data)
        for path in draw_figures(data, detail, load_forecast, pv_forecast):
            print(path)
