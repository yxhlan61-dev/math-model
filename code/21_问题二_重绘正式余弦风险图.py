"""使用正式余弦风险结果覆盖问题二两张调度结果图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from problem2_solver_common import INTERMEDIATE_DIR


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "tmp" / "problem2_cosine7000_band1000_riskadjust"
FIGURE_DIR = ROOT / "figures" / "问题二"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style(ax) -> None:
    ax.grid(True, color="#D0D0D0", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def old_column(frame: pd.DataFrame, text: str) -> str:
    return next(name for name in frame.columns if text in str(name))


def ticks(ax, start: pd.Timestamp) -> None:
    dates = pd.date_range(start.normalize().replace(day=1), "2025-12-01", freq="MS").append(
        pd.DatetimeIndex([pd.Timestamp("2025-12-31")])
    )
    ax.set_xticks(dates)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{mdates.num2date(x).month}.{mdates.num2date(x).day}"))
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlim(start.normalize(), pd.Timestamp("2025-12-31"))


def main() -> None:
    setup(); FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(RISK / "daily.csv", parse_dates=["date"])
    detail = pd.read_csv(RISK / "detail.csv", parse_dates=["date"])
    start = daily["date"].min()
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), sharex=True, constrained_layout=True)
    axes[0].plot(daily["date"], detail.groupby("date")["purchase_kwh"].sum(), color="#1F77B4", linewidth=1.0, label="正常购电量")
    axes[0].fill_between(daily["date"], 0, daily["emergency_kwh"], color="#D62728", alpha=0.40, linewidth=0, label="紧急购电量")
    axes[0].set_ylabel("购电量/(kWh/日)"); axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False); style(axes[0])
    axes[1].plot(daily["date"], daily["end_soc_kwh"], color="#08519C", linewidth=1.25, marker="o", markevery=14, markersize=2.5, label="日末储电量")
    axes[1].plot(daily["date"], daily["band_lower_soc_kwh"], color="#70AD47", linewidth=0.9, linestyle="--", label="微调后SOC下沿")
    axes[1].set_ylabel("日末储电量/kWh"); axes[1].set_xlabel("日期"); axes[1].set_ylim(0, 12000); axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False); style(axes[1]); ticks(axes[1], start)
    fig.savefig(FIGURE_DIR / "问题二_全年正常购电紧急购电与储电量.pdf", bbox_inches="tight"); plt.close(fig)

    no_soc_path = next(INTERMEDIATE_DIR.glob("*无日备惩罚*逐日汇总*.csv"))
    no_soc = pd.read_csv(no_soc_path, encoding="utf-8-sig")
    cdate, ccost, csoc = old_column(no_soc, "日期"), old_column(no_soc, "实际总费用"), old_column(no_soc, "日末储电量")
    no_soc[cdate] = pd.to_datetime(no_soc[cdate]); no_soc = no_soc[no_soc[cdate] >= start]
    months = pd.period_range(start.to_period("M"), "2025-12", freq="M")
    current_month = daily.groupby(daily["date"].dt.to_period("M"))["total_cost_yuan"].sum().reindex(months)
    old_month = no_soc.groupby(no_soc[cdate].dt.to_period("M"))[ccost].sum().reindex(months)
    # 正值代表有 SOC 终端约束的正式方案比无终端 SOC 约束方案少花的钱。
    saving = old_month - current_month
    x = np.arange(len(saving)); labels = [f"{p.month:02d}月" for p in saving.index]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), constrained_layout=True)
    valid = saving.notna().to_numpy()
    axes[0].bar(x[valid], saving.to_numpy()[valid], width=0.66, color="#2C7FB8")
    axes[0].axhline(0, color="#444444", linewidth=0.8); axes[0].set_xticks(x, labels); axes[0].set_ylabel("有 SOC 约束的费用减少量/元"); style(axes[0])
    axes[0].set_ylim(bottom=0)
    axes[1].plot(no_soc[cdate], no_soc[csoc], color="#8C8C8C", linewidth=1.35, drawstyle="steps-post", label="无SOC终端约束")
    axes[1].plot(daily["date"], daily["end_soc_kwh"], color="#08519C", linewidth=1.45, drawstyle="steps-post", label="季节SOC+风险微调")
    axes[1].set_ylabel("日末储电量/kWh"); axes[1].set_xlabel("日期"); axes[1].set_ylim(0, 8000); axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False); style(axes[1]); ticks(axes[1], start)
    fig.savefig(FIGURE_DIR / "问题二_日备不足惩罚前后费用与储能对比.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__": main()
