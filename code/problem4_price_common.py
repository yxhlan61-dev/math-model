"""问题四电价数据、严格因果周期预测、日内残差更新与价格场景。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from problem2_solver_common import N


ROOT = Path(__file__).resolve().parents[1]
ATTACHMENT4 = ROOT / "附件" / "附件4.xlsx"
ISSUE_HOURS = (0, 6, 12, 18)
AR_LAGS = np.asarray([1, 2, 3, 6, 12, 18, 36, 72, 144], dtype=int)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 20.0, 50.0, 100.0)
EXPONENTIAL_RHO = 0.8


@dataclass
class PriceForecastData:
    dates: pd.DatetimeIndex
    actual: np.ndarray
    day_ahead: np.ndarray
    issue_forecast: np.ndarray
    issue_enabled: np.ndarray
    weights: np.ndarray
    ridge_alpha: np.ndarray


def load_actual_prices(expected_dates: pd.DatetimeIndex) -> np.ndarray:
    raw = pd.read_excel(ATTACHMENT4)
    if raw.shape != (365, 145):
        raise ValueError(f"附件4形状应为(365,145)，实际为{raw.shape}")
    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0]))
    if not dates.equals(expected_dates):
        raise ValueError("附件4日期与附件2不一致")
    values = raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or values.min() < 0:
        raise ValueError("附件4包含负电价或非有限值")
    return values


def _exponential_weights(k: int) -> np.ndarray:
    """与问题二、三负荷预测一致的固定指数权重。"""
    values = EXPONENTIAL_RHO ** np.arange(k, dtype=float)
    return values / values.sum()


def causal_day_ahead_forecast(actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    days = actual.shape[0]
    forecast = np.full_like(actual, np.nan)
    weights = np.full((days, 4), np.nan)
    # 首周尚无同星期样本，只在预热期采用上一日同期；正式结果从2月开始。
    for day in range(1, min(7, days)):
        forecast[day] = actual[day - 1]
    for day in range(7, days):
        k = min(4, day // 7)
        current_weights = _exponential_weights(k)
        indices = day - 7 * np.arange(1, k + 1)
        forecast[day] = np.maximum(np.average(actual[indices], axis=0, weights=current_weights), 0.0)
        weights[day, :k] = current_weights
    return forecast, weights


def _ridge_training(residual_flat: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    start = int(AR_LAGS.max())
    indices = np.arange(start, cutoff, dtype=int)
    valid = np.isfinite(residual_flat[indices])
    for lag in AR_LAGS:
        valid &= np.isfinite(residual_flat[indices - lag])
    indices = indices[valid]
    if len(indices) < 7 * N:
        return np.empty((0, len(AR_LAGS))), np.empty(0)
    features = np.column_stack([residual_flat[indices - lag] for lag in AR_LAGS])
    return features, residual_flat[indices]


def _fit_causal_ridge(features: np.ndarray, target: np.ndarray) -> tuple[Ridge, float]:
    split = max(int(len(target) * 0.8), len(target) - 14 * N)
    split = min(max(split, N), len(target) - N)
    best_alpha = 20.0
    best_mae = np.inf
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(features[:split], target[:split])
        mae = float(np.mean(np.abs(model.predict(features[split:]) - target[split:])))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return Ridge(alpha=best_alpha).fit(features, target), best_alpha


def build_price_forecasts(dates: pd.DatetimeIndex) -> PriceForecastData:
    actual = load_actual_prices(dates)
    base, weights = causal_day_ahead_forecast(actual)
    issue_forecast = np.repeat(base[:, None, :], len(ISSUE_HOURS), axis=1)
    enabled = np.zeros((len(dates), len(ISSUE_HOURS)), dtype=bool)
    alpha_used = np.full((len(dates), len(ISSUE_HOURS)), np.nan)
    residual = actual - base
    flat = residual.ravel()

    enabled[7:, 0] = True
    # 每月首日只用此前数据选参并拟合；月内固定系数，仍是严格因果且显著减少重复拟合。
    for month in range(2, 13):
        month_days = np.where(dates.month == month)[0]
        cutoff_day = int(month_days[0])
        for issue_index, issue_hour in enumerate(ISSUE_HOURS[1:], start=1):
            start = issue_hour * 6
            validation_days = np.arange(max(7, cutoff_day - 7), cutoff_day, dtype=int)
            gate_features, gate_target = _ridge_training(flat, int(validation_days[0]) * N)
            use_update = False
            if len(gate_target):
                gate_model, _ = _fit_causal_ridge(gate_features, gate_target)
                candidates = []
                for validation_day in validation_days:
                    work = flat.copy()
                    work[validation_day * N + start:(validation_day + 1) * N] = np.nan
                    for index in range(validation_day * N + start, (validation_day + 1) * N):
                        values = np.asarray([work[index - lag] for lag in AR_LAGS], dtype=float)
                        work[index] = float(gate_model.predict(values[None, :])[0])
                    candidates.append(np.maximum(
                        base[validation_day, start:] +
                        work[validation_day * N + start:(validation_day + 1) * N], 0.0
                    ))
                candidate_array = np.asarray(candidates)
                base_mae = np.abs(actual[validation_days, start:] - base[validation_days, start:]).mean()
                update_mae = np.abs(actual[validation_days, start:] - candidate_array).mean()
                use_update = bool(update_mae < base_mae)

            features, target = _ridge_training(flat, cutoff_day * N)
            if len(target) == 0:
                continue
            model, alpha = _fit_causal_ridge(features, target)
            for day in month_days:
                alpha_used[day, issue_index] = alpha
                work = flat.copy()
                work[day * N + start:(day + 1) * N] = np.nan
                for index in range(day * N + start, (day + 1) * N):
                    values = np.asarray([work[index - lag] for lag in AR_LAGS], dtype=float)
                    work[index] = float(model.predict(values[None, :])[0])
                candidate = np.maximum(base[day, start:] + work[day * N + start:(day + 1) * N], 0.0)
                if use_update:
                    issue_forecast[day, issue_index, start:] = candidate
                    enabled[day, issue_index] = True

    return PriceForecastData(dates, actual, base, issue_forecast, enabled, weights, alpha_used)


def price_scenarios(
    forecast_data: PriceForecastData,
    day: int,
    issue_hour: int,
    start: int,
    sampled_days: list[int],
) -> np.ndarray:
    issue_index = ISSUE_HOURS.index(issue_hour)
    point = forecast_data.issue_forecast[day, issue_index, start:]
    if not np.isfinite(point).all():
        raise ValueError("当前价格预测含缺失值")
    if not sampled_days:
        return point[None, :]
    sampled = np.asarray(sampled_days, dtype=int)
    historical_forecast = forecast_data.issue_forecast[sampled, issue_index, start:]
    if not np.isfinite(historical_forecast).all():
        raise ValueError("历史价格预测残差不完整")
    residual = forecast_data.actual[sampled, start:] - historical_forecast
    return np.maximum(point[None, :] + residual, 0.0)


def reserve_penalty(actual: np.ndarray, day: int, eta_d: float = 0.9) -> float:
    if day <= 0:
        return 0.0
    return float(eta_d * np.quantile(actual[:day], 0.90))


def forecast_metrics(data: PriceForecastData, start_day: int = 31) -> dict:
    from scipy.stats import spearmanr

    actual = data.actual[start_day:]
    models = {
        "上一日同期": data.actual[start_day - 1:-1],
        "上一周同期": data.actual[start_day - 7:-7],
        "最近4周简单平均": np.asarray([
            data.actual[d - 7 * np.arange(1, 5)].mean(axis=0) for d in range(start_day, len(data.dates))
        ]),
        "最近4周指数加权（正式）": data.day_ahead[start_day:],
    }
    result = {}
    for name, prediction in models.items():
        error = prediction - actual
        daily_rank = [spearmanr(prediction[i], actual[i]).statistic for i in range(len(actual))]
        high_count = int(np.ceil(N * 0.20))
        true_high = np.argpartition(actual, -high_count, axis=1)[:, -high_count:]
        pred_high = np.argpartition(prediction, -high_count, axis=1)[:, -high_count:]
        overlap = np.mean([
            len(set(true_high[i]).intersection(pred_high[i])) / high_count for i in range(len(actual))
        ])
        result[name] = {
            "MAE(元/kWh)": float(np.mean(np.abs(error))),
            "RMSE(元/kWh)": float(np.sqrt(np.mean(error ** 2))),
            "WAPE": float(np.sum(np.abs(error)) / np.sum(actual)),
            "日内Spearman": float(np.nanmean(daily_rank)),
            "前20%高价时段识别精确率": float(overlap),
            "最高价时刻MAE(分钟)": float(np.mean(np.abs(
                np.argmax(prediction, axis=1) - np.argmax(actual, axis=1)
            )) * 10.0),
        }
    updates = {}
    for issue_index, issue_hour in enumerate(ISSUE_HOURS[1:], start=1):
        start = issue_hour * 6
        base_error = np.abs(data.actual[start_day:, start:] - data.day_ahead[start_day:, start:])
        update_error = np.abs(
            data.actual[start_day:, start:] - data.issue_forecast[start_day:, issue_index, start:]
        )
        updates[str(issue_hour)] = {
            "采用更新天数": int(data.issue_enabled[start_day:, issue_index].sum()),
            "周期基线MAE(元/kWh)": float(base_error.mean()),
            "门控更新MAE(元/kWh)": float(update_error.mean()),
        }
    flattened = data.actual.ravel()
    centered = flattened - flattened.mean()
    denominator = float(np.dot(centered, centered))
    acf_daily = {
        str(lag_day): float(np.dot(centered[lag_day * N:], centered[:-lag_day * N]) / denominator)
        for lag_day in range(1, 29)
    }
    return {"0点模型比较": result, "日内更新": updates, "ACF整日滞后": acf_daily}
