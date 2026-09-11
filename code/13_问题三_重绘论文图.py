"""读取问题三已保存的全年结果，更新预测指标并重绘论文图。"""

from __future__ import annotations

import json

import pandas as pd

from problem3_solver_common import INTERMEDIATE_DIR, load_problem3_data, write_json
from importlib import import_module


def main() -> None:
    module = import_module("11_问题三_四时点滚动MILP")
    data = load_problem3_data()
    all_daily = pd.read_csv(INTERMEDIATE_DIR / "问题三_四策略逐日汇总.csv", parse_dates=["日期"])
    official_detail = pd.read_csv(
        INTERMEDIATE_DIR / "S61218_全部四时点_逐时段结果.csv", parse_dates=["日期"]
    )
    forecasts = module.forecast_metrics(data)
    module.draw_figures(data, all_daily, official_detail, forecasts)

    summary_path = INTERMEDIATE_DIR / "问题三_汇总与校验.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["光伏预测指标"] = forecasts
    write_json(summary_path, summary)
    bundle_path = INTERMEDIATE_DIR / "问题三_工作簿数据.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["forecastMetrics"] = forecasts
    write_json(bundle_path, bundle)
    print("问题三论文图与同目标预测指标已更新")


if __name__ == "__main__":
    main()
