from __future__ import annotations

import numpy as np
import pandas as pd


GAS_DECAY_PER_HOUR = {
    "CO": 0.01,
    "NO": 0.18,
    "NO2": 0.08,
    "O3": 0.04,
    "SO2": 0.07,
}

GAS_RAIN_SCAVENGING = {
    "CO": 0.002,
    "NO": 0.004,
    "NO2": 0.012,
    "O3": 0.004,
    "SO2": 0.025,
}


def _numeric(frame: pd.DataFrame, column: str, default: float | pd.Series) -> pd.Series:
    if column not in frame:
        if isinstance(default, pd.Series):
            return default.astype(float)
        return pd.Series(float(default), index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    if isinstance(default, pd.Series):
        return values.fillna(default)
    return values.fillna(float(default))


def add_physical_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent one-hour transport baseline at station locations.

    CatBoost learns only the residual left by this baseline. The gridded
    forecast uses the same decay/rain coefficients in the transport solver.
    """
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    baselines = pd.Series(np.nan, index=result.index, dtype=float)
    for substance_code, indexes in result.groupby("substance_code", sort=False).groups.items():
        code = str(substance_code)
        local = _numeric(result.loc[indexes], f"gas_{code}_lag_0", np.nan).clip(lower=0.0)
        neighbor = _numeric(result.loc[indexes], "spatial_upwind_value", local).clip(lower=0.0)
        wind = _numeric(result.loc[indexes], "transport_weather_hor_win_spd", 0.0).clip(lower=0.0)
        rain = _numeric(result.loc[indexes], "transport_weather_precipitation", 0.0).clip(lower=0.0)
        source_pbl = _numeric(
            result.loc[indexes], "source_weather_boundary_layer_height", 500.0
        ).clip(50.0, 4000.0)
        target_pbl = _numeric(
            result.loc[indexes], "target_weather_boundary_layer_height", source_pbl
        ).clip(50.0, 4000.0)

        neighbor_blend = (wind / 12.0).clip(0.0, 0.35)
        transported = local * (1.0 - neighbor_blend) + neighbor * neighbor_blend
        vertical_mixing = np.sqrt(source_pbl / target_pbl).clip(0.45, 1.8)
        attenuation = np.exp(
            -GAS_DECAY_PER_HOUR.get(code, 0.05)
            -GAS_RAIN_SCAVENGING.get(code, 0.008) * rain
        )
        baselines.loc[indexes] = (transported * vertical_mixing * attenuation).clip(lower=0.0)
    result["physical_baseline"] = baselines
    return result
