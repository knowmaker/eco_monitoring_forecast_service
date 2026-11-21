from __future__ import annotations

from math import asin, atan2, cos, degrees, exp, radians, sin, sqrt

import numpy as np
import pandas as pd

from app.config import get_settings


SPATIAL_COLUMNS = [
    "spatial_upwind_value",
    "spatial_neighbor_mean",
    "spatial_neighbor_max",
    "spatial_weight_sum",
    "spatial_neighbor_count",
    "spatial_min_distance_km",
    "backtrajectory_value",
    "backtrajectory_distance_km",
    "backtrajectory_travel_hours",
    "backtrajectory_alignment",
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
    locations = {
        int(post_id): (float(location.latitude), float(location.longitude))
        for post_id, location in post_locations.iterrows()
        if np.isfinite(location.latitude) and np.isfinite(location.longitude)
    }
    geometry = {
        target_id: [
            (neighbor_id, *_distance_and_bearing(*target, *neighbor))
            for neighbor_id, neighbor in locations.items()
            if neighbor_id != target_id
        ]
        for target_id, target in locations.items()
    }
    lag_deltas = [pd.Timedelta(hours=hour) for hour in range(4)]
    influence_meters = get_settings().MAX_OBSERVATION_INFLUENCE_METERS

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
        trajectory_values: list[tuple[float, float, float, float, float]] = []

        if all(np.isfinite(item) for item in (wind_direction, wind_speed, target_latitude, target_longitude)):
            for neighbor_id, distance_km, bearing_to_neighbor in geometry[target_id]:
                alignment = _angular_alignment(float(wind_direction), bearing_to_neighbor)
                if alignment <= 0:
                    continue
                travel_hours = distance_km / max(float(wind_speed) * 3.6, 0.1)
                lag_hours = int(np.clip(round(travel_hours), 0, 3))
                value = lookup.get((neighbor_id, row.source_bucket - lag_deltas[lag_hours]))
                if value is None or not np.isfinite(value):
                    continue
                weight = alignment * exp(-distance_km / 20.0)
                values.append(float(value))
                weighted_values.append((float(value), weight))
                distances.append(distance_km)
                if distance_km * 1000.0 <= influence_meters:
                    trajectory_weight = alignment * exp(
                        -0.5
                        * (distance_km * 1000.0 / influence_meters) ** 2
                    )
                    trajectory_values.append(
                        (float(value), trajectory_weight, distance_km, travel_hours, alignment)
                    )

        weight_sum = sum(weight for _, weight in weighted_values)
        trajectory_weight_sum = sum(item[1] for item in trajectory_values)
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
                "backtrajectory_value": (
                    sum(value * weight for value, weight, _, _, _ in trajectory_values)
                    / trajectory_weight_sum
                    if trajectory_weight_sum > 0
                    else np.nan
                ),
                "backtrajectory_distance_km": (
                    sum(distance * weight for _, weight, distance, _, _ in trajectory_values)
                    / trajectory_weight_sum
                    if trajectory_weight_sum > 0
                    else np.nan
                ),
                "backtrajectory_travel_hours": (
                    sum(hours * weight for _, weight, _, hours, _ in trajectory_values)
                    / trajectory_weight_sum
                    if trajectory_weight_sum > 0
                    else np.nan
                ),
                "backtrajectory_alignment": (
                    sum(alignment * weight for _, weight, _, _, alignment in trajectory_values)
                    / trajectory_weight_sum
                    if trajectory_weight_sum > 0
                    else np.nan
                ),
            }
        )

    spatial = pd.DataFrame(calculated, index=result.index)
    for column in SPATIAL_COLUMNS:
        result[column] = spatial[column]
    return result
