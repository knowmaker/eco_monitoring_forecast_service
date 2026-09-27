from __future__ import annotations

from math import asin, atan2, cos, degrees, exp, radians, sin, sqrt

import numpy as np
import pandas as pd


SPATIAL_COLUMNS = [
    "spatial_upwind_value",
    "spatial_neighbor_mean",
    "spatial_neighbor_max",
    "spatial_weight_sum",
    "spatial_neighbor_count",
    "spatial_min_distance_km",
]


def _distance_and_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    radius_km = 6371.0088
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    distance = 2 * radius_km * asin(min(1.0, sqrt(a)))
    y = sin(delta_lambda) * cos(phi2)
    x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(delta_lambda)
    bearing = (degrees(atan2(y, x)) + 360.0) % 360.0
    return distance, bearing


def _angular_alignment(wind_from_deg: float, neighbor_bearing_deg: float) -> float:
    difference = radians((wind_from_deg - neighbor_bearing_deg + 180.0) % 360.0 - 180.0)
    return max(0.0, cos(difference))


def add_spatial_features(frame: pd.DataFrame, target_substance: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    value_column = f"gas_{target_substance.upper()}_lag_0"
    lookup = {
        (int(row.monitoring_post_id), row.source_bucket): getattr(row, value_column)
        for row in result.itertuples(index=False)
    }
    post_locations = (
        result[["monitoring_post_id", "latitude", "longitude"]]
        .drop_duplicates("monitoring_post_id")
        .set_index("monitoring_post_id")
    )

    calculated: list[dict[str, float]] = []
    for row in result.itertuples(index=False):
        target_id = int(row.monitoring_post_id)
        wind_direction = getattr(row, "source_weather_hor_win_dir", np.nan)
        wind_speed = getattr(row, "source_weather_hor_win_spd", np.nan)
        target_latitude = getattr(row, "latitude", np.nan)
        target_longitude = getattr(row, "longitude", np.nan)
        values: list[float] = []
        weighted_values: list[tuple[float, float]] = []
        distances: list[float] = []

        if all(np.isfinite(item) for item in (wind_direction, wind_speed, target_latitude, target_longitude)):
            for neighbor_id, location in post_locations.iterrows():
                neighbor_id = int(neighbor_id)
                if neighbor_id == target_id or not np.isfinite(location.latitude) or not np.isfinite(location.longitude):
                    continue
                distance_km, bearing_to_neighbor = _distance_and_bearing(
                    float(target_latitude),
                    float(target_longitude),
                    float(location.latitude),
                    float(location.longitude),
                )
                alignment = _angular_alignment(float(wind_direction), bearing_to_neighbor)
                if alignment <= 0:
                    continue
                travel_hours = distance_km / max(float(wind_speed) * 3.6, 0.1)
                lag_hours = int(np.clip(round(travel_hours), 0, 3))
                value = lookup.get((neighbor_id, row.source_bucket - pd.Timedelta(hours=lag_hours)))
                if value is None or not np.isfinite(value):
                    continue
                weight = alignment * exp(-distance_km / 20.0)
                values.append(float(value))
                weighted_values.append((float(value), weight))
                distances.append(distance_km)

        weight_sum = sum(weight for _, weight in weighted_values)
        calculated.append(
            {
                "spatial_upwind_value": (
                    sum(value * weight for value, weight in weighted_values) / weight_sum if weight_sum > 0 else np.nan
                ),
                "spatial_neighbor_mean": float(np.mean(values)) if values else np.nan,
                "spatial_neighbor_max": float(np.max(values)) if values else np.nan,
                "spatial_weight_sum": weight_sum if weighted_values else 0.0,
                "spatial_neighbor_count": float(len(values)),
                "spatial_min_distance_km": float(min(distances)) if distances else np.nan,
            }
        )

    spatial = pd.DataFrame(calculated, index=result.index)
    for column in SPATIAL_COLUMNS:
        result[column] = spatial[column]
    return result
