from __future__ import annotations

from math import cos, pi

import numpy as np
import pandas as pd

from app.config import get_settings


METERS_PER_DEGREE_LATITUDE = 111_320.0


def _project(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    reference_latitude: float,
    reference_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    longitude_scale = METERS_PER_DEGREE_LATITUDE * cos(reference_latitude * pi / 180.0)
    x = (longitudes - reference_longitude) * longitude_scale
    y = (latitudes - reference_latitude) * METERS_PER_DEGREE_LATITUDE
    return x, y


def _unproject(
    x: np.ndarray,
    y: np.ndarray,
    reference_latitude: float,
    reference_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    longitude_scale = METERS_PER_DEGREE_LATITUDE * cos(reference_latitude * pi / 180.0)
    latitudes = reference_latitude + y / METERS_PER_DEGREE_LATITUDE
    longitudes = reference_longitude + x / longitude_scale
    return latitudes, longitudes


def _station_clusters(x: np.ndarray, y: np.ndarray, maximum_distance: float) -> list[list[int]]:
    parents = list(range(len(x)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(x)):
        for right in range(left + 1, len(x)):
            if float(np.hypot(x[left] - x[right], y[left] - y[right])) <= maximum_distance:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(x)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _wind_components(wind_speed: float | None, wind_from_degrees: float | None) -> tuple[float, float]:
    if wind_speed is None or wind_from_degrees is None:
        return 0.0, 0.0
    if not np.isfinite(wind_speed) or not np.isfinite(wind_from_degrees):
        return 0.0, 0.0
    wind_to_radians = np.deg2rad((float(wind_from_degrees) + 180.0) % 360.0)
    return float(np.sin(wind_to_radians)), float(np.cos(wind_to_radians))


def _advection_distance(wind_speed: float | None, maximum_distance: float) -> float:
    if wind_speed is None or not np.isfinite(wind_speed):
        return 0.0
    return min(max(float(wind_speed), 0.0) * 3600.0, maximum_distance)


def _interpolation_weights(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    station_x: np.ndarray,
    station_y: np.ndarray,
    wind_speed: float | None,
    wind_from_degrees: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    delta_x = grid_x[:, None] - station_x[None, :]
    delta_y = grid_y[:, None] - station_y[None, :]
    distances = np.hypot(delta_x, delta_y)
    wind_x, wind_y = _wind_components(wind_speed, wind_from_degrees)
    if wind_x == 0.0 and wind_y == 0.0:
        scale = 450.0
        weights = np.exp(-0.5 * (distances / scale) ** 2)
    else:
        along_wind = delta_x * wind_x + delta_y * wind_y
        across_wind = -delta_x * wind_y + delta_y * wind_x
        downwind_scale = 450.0 + min(max(float(wind_speed), 0.0), 12.0) * 90.0
        along_scale = np.where(along_wind >= 0.0, downwind_scale, 300.0)
        weights = np.exp(-0.5 * ((along_wind / along_scale) ** 2 + (across_wind / 350.0) ** 2))
    weights *= 1.0 / np.maximum(distances, 25.0)
    return weights, distances


def _weighted_values(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(weights.shape[0], np.nan)
    usable = weights[:, finite]
    totals = usable.sum(axis=1)
    return np.divide(
        usable @ values[finite],
        totals,
        out=np.full(weights.shape[0], np.nan),
        where=totals > 0,
    )


def _confidence(distances: np.ndarray, station_count: int) -> np.ndarray:
    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    supporting_distance = ordered[:, min(2, station_count - 1)]
    buffer = float(get_settings().GRID_BUFFER_METERS)
    proximity = np.exp(-0.5 * (nearest / (buffer * 0.55)) ** 2)
    geometry = (
        np.exp(-supporting_distance / (buffer * 1.5))
        if station_count > 1
        else np.ones_like(nearest)
    )
    return np.clip(proximity * geometry, 0.0, 1.0)


def _correct_to_anchors(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    values: np.ndarray,
    station_x: np.ndarray,
    station_y: np.ndarray,
    station_values: np.ndarray,
    wind_speed: float | None,
    wind_from_degrees: float | None,
    estimated_at_stations: np.ndarray | None = None,
) -> np.ndarray:
    if estimated_at_stations is None:
        station_weights, _ = _interpolation_weights(
            station_x,
            station_y,
            station_x,
            station_y,
            wind_speed,
            wind_from_degrees,
        )
        estimated_at_stations = _weighted_values(station_weights, station_values)
    residuals = station_values - estimated_at_stations
    delta_x = grid_x[:, None] - station_x[None, :]
    delta_y = grid_y[:, None] - station_y[None, :]
    correction_weights = np.exp(-0.5 * (np.hypot(delta_x, delta_y) / 180.0) ** 2)
    return values + _weighted_values(correction_weights, residuals)


def _interpolate_cluster(
    anchors: pd.DataFrame,
    source_anchors: pd.DataFrame,
    *,
    cluster_id: int,
    data_kind: str,
    wind_speed: float | None,
    wind_from_degrees: float | None,
) -> pd.DataFrame:
    settings = get_settings()
    reference_latitude = float(anchors["latitude"].mean())
    reference_longitude = float(anchors["longitude"].mean())
    station_x, station_y = _project(
        anchors["latitude"].to_numpy(dtype=float),
        anchors["longitude"].to_numpy(dtype=float),
        reference_latitude,
        reference_longitude,
    )
    step = float(settings.GRID_CELL_METERS)
    buffer = float(settings.GRID_BUFFER_METERS)
    transport_anchors = source_anchors
    if data_kind == "forecast" and len(transport_anchors) < settings.GRID_MIN_STATIONS:
        transport_anchors = anchors
    source_x = np.asarray([], dtype=float)
    source_y = np.asarray([], dtype=float)
    wind_x, wind_y = _wind_components(wind_speed, wind_from_degrees)
    displacement = 0.0
    if data_kind == "forecast" and len(transport_anchors) >= settings.GRID_MIN_STATIONS:
        source_x, source_y = _project(
            transport_anchors["latitude"].to_numpy(dtype=float),
            transport_anchors["longitude"].to_numpy(dtype=float),
            reference_latitude,
            reference_longitude,
        )
        displacement = _advection_distance(
            wind_speed,
            float(settings.GRID_MAX_ADVECTION_METERS),
        )

    coverage_x = np.concatenate((station_x, source_x + wind_x * displacement))
    coverage_y = np.concatenate((station_y, source_y + wind_y * displacement))
    x_values = np.arange(
        np.floor((coverage_x.min() - buffer) / step) * step,
        np.ceil((coverage_x.max() + buffer) / step) * step + step,
        step,
    )
    y_values = np.arange(
        np.floor((coverage_y.min() - buffer) / step) * step,
        np.ceil((coverage_y.max() + buffer) / step) * step + step,
        step,
    )
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    grid_x = mesh_x.ravel()
    grid_y = mesh_y.ravel()
    weights, distances = _interpolation_weights(
        grid_x,
        grid_y,
        station_x,
        station_y,
        wind_speed,
        wind_from_degrees,
    )
    station_values = anchors["value"].to_numpy(dtype=float)
    values = _weighted_values(weights, station_values)
    station_weights, _ = _interpolation_weights(
        station_x,
        station_y,
        station_x,
        station_y,
        wind_speed,
        wind_from_degrees,
    )
    estimated_at_stations = _weighted_values(station_weights, station_values)
    transported_confidence = np.zeros(len(grid_x), dtype=float)

    if source_x.size:
        source_weights, source_distances = _interpolation_weights(
            grid_x - wind_x * displacement,
            grid_y - wind_y * displacement,
            source_x,
            source_y,
            wind_speed,
            wind_from_degrees,
        )
        transported = _weighted_values(source_weights, transport_anchors["value"].to_numpy(dtype=float))
        usable = np.isfinite(transported)
        values[usable] = (
            values[usable] * (1.0 - settings.GRID_TRANSPORT_BLEND)
            + transported[usable] * settings.GRID_TRANSPORT_BLEND
        )
        source_station_weights, _ = _interpolation_weights(
            station_x - wind_x * displacement,
            station_y - wind_y * displacement,
            source_x,
            source_y,
            wind_speed,
            wind_from_degrees,
        )
        transported_at_stations = _weighted_values(
            source_station_weights,
            transport_anchors["value"].to_numpy(dtype=float),
        )
        usable_at_stations = np.isfinite(transported_at_stations)
        estimated_at_stations[usable_at_stations] = (
            estimated_at_stations[usable_at_stations] * (1.0 - settings.GRID_TRANSPORT_BLEND)
            + transported_at_stations[usable_at_stations] * settings.GRID_TRANSPORT_BLEND
        )
        transported_confidence = 0.7 * _confidence(
            source_distances,
            len(transport_anchors),
        )

    values = _correct_to_anchors(
        grid_x,
        grid_y,
        values,
        station_x,
        station_y,
        station_values,
        wind_speed,
        wind_from_degrees,
        estimated_at_stations,
    )
    if len(anchors) == 1:
        values.fill(station_values[0])
    values = np.maximum(values, 0.0)

    lower_bounds = _weighted_values(weights, anchors["lower_bound"].to_numpy(dtype=float))
    upper_bounds = _weighted_values(weights, anchors["upper_bound"].to_numpy(dtype=float))
    lower_bounds = np.where(np.isfinite(lower_bounds), np.minimum(lower_bounds, values), np.nan)
    upper_bounds = np.where(np.isfinite(upper_bounds), np.maximum(upper_bounds, values), np.nan)
    confidence = np.maximum(
        _confidence(distances, len(anchors)),
        transported_confidence,
    )
    keep = np.isfinite(values) & (confidence >= 0.04)
    grid_x = grid_x[keep]
    grid_y = grid_y[keep]
    values = values[keep]
    confidence = confidence[keep]
    lower_bounds = lower_bounds[keep]
    upper_bounds = upper_bounds[keep]

    latitudes, longitudes = _unproject(
        grid_x,
        grid_y,
        reference_latitude,
        reference_longitude,
    )
    south, west = _unproject(
        grid_x - step / 2.0,
        grid_y - step / 2.0,
        reference_latitude,
        reference_longitude,
    )
    north, east = _unproject(
        grid_x + step / 2.0,
        grid_y + step / 2.0,
        reference_latitude,
        reference_longitude,
    )
    x_indexes = np.searchsorted(x_values, grid_x)
    y_indexes = np.searchsorted(y_values, grid_y)
    return pd.DataFrame(
        {
            "cluster_id": cluster_id,
            "grid_x": x_indexes,
            "grid_y": y_indexes,
            "latitude": latitudes,
            "longitude": longitudes,
            "south": south,
            "west": west,
            "north": north,
            "east": east,
            "value": values,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
            "confidence": confidence,
            "source_station_count": len(anchors),
            "wind_speed": wind_speed,
            "wind_direction": wind_from_degrees,
        }
    )


def build_concentration_grid(
    anchors: pd.DataFrame,
    *,
    data_kind: str,
    wind_speed: float | None = None,
    wind_from_degrees: float | None = None,
    source_anchors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"monitoring_post_id", "latitude", "longitude", "value"}
    if anchors.empty or not required.issubset(anchors.columns):
        return pd.DataFrame()
    clean = anchors.copy()
    for column in ("latitude", "longitude", "value", "lower_bound", "upper_bound"):
        if column not in clean:
            clean[column] = np.nan
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["latitude", "longitude", "value"]).reset_index(drop=True)
    clean["value"] = clean["value"].abs()
    valid_interval = clean["lower_bound"].notna() & clean["upper_bound"].notna()
    interval_crosses_zero = (
        valid_interval
        & (clean["lower_bound"] <= 0)
        & (clean["upper_bound"] >= 0)
    )
    absolute_lower = np.minimum(clean["lower_bound"].abs(), clean["upper_bound"].abs())
    absolute_upper = np.maximum(clean["lower_bound"].abs(), clean["upper_bound"].abs())
    clean.loc[valid_interval, "lower_bound"] = absolute_lower[valid_interval]
    clean.loc[interval_crosses_zero, "lower_bound"] = 0.0
    clean.loc[valid_interval, "upper_bound"] = absolute_upper[valid_interval]
    if len(clean) < get_settings().GRID_MIN_STATIONS:
        return pd.DataFrame()

    reference_latitude = float(clean["latitude"].mean())
    reference_longitude = float(clean["longitude"].mean())
    x, y = _project(
        clean["latitude"].to_numpy(dtype=float),
        clean["longitude"].to_numpy(dtype=float),
        reference_latitude,
        reference_longitude,
    )
    source = source_anchors.copy() if source_anchors is not None else pd.DataFrame()
    if not source.empty:
        for column in ("latitude", "longitude", "value"):
            if column not in source:
                source[column] = np.nan
            source[column] = pd.to_numeric(source[column], errors="coerce")
        source = source.dropna(subset=["monitoring_post_id", "latitude", "longitude", "value"])
        source["value"] = source["value"].abs()
    frames: list[pd.DataFrame] = []
    for indexes in _station_clusters(x, y, float(get_settings().GRID_CLUSTER_DISTANCE_METERS)):
        if len(indexes) < get_settings().GRID_MIN_STATIONS:
            continue
        cluster = clean.iloc[indexes].reset_index(drop=True)
        station_ids = set(cluster["monitoring_post_id"].astype(int))
        cluster_source = source[source["monitoring_post_id"].astype(int).isin(station_ids)].copy() if not source.empty else source
        frames.append(
            _interpolate_cluster(
                cluster,
                cluster_source,
                cluster_id=int(cluster["monitoring_post_id"].min()),
                data_kind=data_kind,
                wind_speed=wind_speed,
                wind_from_degrees=wind_from_degrees,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
