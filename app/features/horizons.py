from __future__ import annotations

import pandas as pd

from app.features.local_features import _add_time_features
from app.features.weather_features import WEATHER_COLUMNS


FORECAST_HORIZONS = (1, 2, 3, 4, 5)


def expand_horizons(
    source_frame: pd.DataFrame,
    target_weather: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    include_target: bool,
) -> pd.DataFrame:
    """Reuse causal source features for each target hour after the source bucket."""
    if source_frame.empty:
        return source_frame.copy()
    if not horizons or any(hour < 1 for hour in horizons):
        raise ValueError("Horizons must be positive hours from the source bucket")

    target_columns = [f"target_weather_{column}" for column in WEATHER_COLUMNS]
    source = source_frame.drop(
        columns=[
            "target_start", "target_end", "target_value", "correction_target",
            "physical_baseline", "horizon_hours", "target_hour_sin", "target_hour_cos",
            "target_weekday_sin", "target_weekday_cos", *target_columns,
        ],
        errors="ignore",
    )
    weather = target_weather.rename(
        columns={
            "bucket_start": "target_start",
            **{column: f"target_weather_{column}" for column in WEATHER_COLUMNS},
        }
    )
    if include_target:
        codes = source["substance_code"].unique()
        if len(codes) != 1:
            raise ValueError("Target lookup requires exactly one gas per source frame")
        current_column = f"gas_{codes[0]}_lag_0"
        labels = source[["monitoring_post_id", "source_bucket", current_column]].rename(
            columns={"source_bucket": "target_start", current_column: "target_value"}
        )

    frames = []
    for hour in horizons:
        frame = source.copy()
        frame["horizon_hours"] = hour
        frame["target_start"] = frame["source_bucket"] + pd.Timedelta(hours=hour)
        frame["target_end"] = frame["target_start"] + pd.Timedelta(hours=1)
        frame = frame.merge(
            weather, on=["monitoring_post_id", "target_start"], how="left",
            validate="many_to_one",
        )
        _add_time_features(frame)
        if include_target:
            frame = frame.merge(
                labels, on=["monitoring_post_id", "target_start"], how="left",
                validate="many_to_one",
            )
            frame = frame[frame[current_column].notna() & frame["target_value"].notna()]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)
