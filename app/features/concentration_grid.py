from __future__ import annotations

from math import cos, pi

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from app.config import get_settings
from app.features.physical_baseline import GAS_DECAY_PER_HOUR, GAS_RAIN_SCAVENGING


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
    inside_coverage = distances.min(axis=1) <= buffer
    if source_x.size:
        transported_distances = np.hypot(
            grid_x[:, None] - (source_x + wind_x * displacement)[None, :],
            grid_y[:, None] - (source_y + wind_y * displacement)[None, :],
        )
        inside_coverage |= transported_distances.min(axis=1) <= buffer
    keep = np.isfinite(values) & (confidence >= 0.04) & inside_coverage
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
    clean["value"] = clean["value"].clip(lower=0.0)
    valid_interval = clean["lower_bound"].notna() & clean["upper_bound"].notna()
    clean.loc[valid_interval, "lower_bound"] = clean.loc[valid_interval, "lower_bound"].clip(lower=0.0)
    clean.loc[valid_interval, "upper_bound"] = clean.loc[valid_interval, "upper_bound"].clip(lower=0.0)
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
        source["value"] = source["value"].clip(lower=0.0)
    frames: list[pd.DataFrame] = []
    for indexes in _station_clusters(x, y, float(get_settings().GRID_CLUSTER_DISTANCE_METERS)):
        if len(indexes) < get_settings().GRID_MIN_STATIONS:
            continue
        cluster = clean.iloc[indexes].reset_index(drop=True)
        station_ids = set(cluster["monitoring_post_id"].astype(int))
        cluster_source = (
            source[source["monitoring_post_id"].astype(int).isin(station_ids)].copy()
            if not source.empty
            else source
        )
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


def assimilate_observations(
    anchors: pd.DataFrame,
    prior_grid: pd.DataFrame | None,
    *,
    wind_speed: float | None = None,
    wind_from_degrees: float | None = None,
) -> pd.DataFrame:
    """Correct the previous forecast with current station observations."""
    analysis = build_concentration_grid(
        anchors,
        data_kind="observed",
        wind_speed=wind_speed,
        wind_from_degrees=wind_from_degrees,
    )
    if analysis.empty:
        return analysis
    analysis["analysis_value"] = analysis["value"]
    analysis["physical_forecast"] = np.nan
    analysis["correction_value"] = 0.0
    analysis["diffusion_coefficient"] = np.nan
    analysis["decay_coefficient"] = np.nan
    wind_x, wind_y = _wind_components(wind_speed, wind_from_degrees)
    analysis["wind_u"] = wind_x * (float(wind_speed) if wind_speed is not None else 0.0)
    analysis["wind_v"] = wind_y * (float(wind_speed) if wind_speed is not None else 0.0)
    analysis["boundary_layer_height"] = np.nan

    if prior_grid is None or prior_grid.empty:
        return analysis

    radius = float(get_settings().PHYSICS_ASSIMILATION_RADIUS_METERS)
    for cluster_id, indexes in analysis.groupby("cluster_id", sort=False).groups.items():
        output = analysis.loc[indexes]
        cluster_prior = prior_grid[prior_grid["cluster_id"] == cluster_id]
        if cluster_prior.empty:
            continue
        reference_latitude = float(output["latitude"].mean())
        reference_longitude = float(output["longitude"].mean())
        output_x, output_y = _project(
            output["latitude"].to_numpy(float), output["longitude"].to_numpy(float),
            reference_latitude, reference_longitude,
        )
        prior_x, prior_y = _project(
            cluster_prior["latitude"].to_numpy(float), cluster_prior["longitude"].to_numpy(float),
            reference_latitude, reference_longitude,
        )
        prior_tree = cKDTree(np.column_stack((prior_x, prior_y)))
        prior_distance, prior_index = prior_tree.query(np.column_stack((output_x, output_y)), k=1)
        usable_prior = prior_distance <= float(get_settings().GRID_CELL_METERS) * 1.6
        background = output["value"].to_numpy(float)
        prior_values = cluster_prior["value"].to_numpy(float)[prior_index]
        background[usable_prior] = prior_values[usable_prior]

        all_anchor_x, all_anchor_y = _project(
            anchors["latitude"].to_numpy(float), anchors["longitude"].to_numpy(float),
            reference_latitude, reference_longitude,
        )
        near_cluster = np.hypot(all_anchor_x, all_anchor_y) <= (
            float(get_settings().GRID_CLUSTER_DISTANCE_METERS)
            + float(get_settings().GRID_BUFFER_METERS)
        )
        cluster_anchors = anchors.loc[near_cluster]
        if cluster_anchors.empty:
            continue
        station_x, station_y = _project(
            cluster_anchors["latitude"].to_numpy(float), cluster_anchors["longitude"].to_numpy(float),
            reference_latitude, reference_longitude,
        )
        _, station_prior_index = prior_tree.query(np.column_stack((station_x, station_y)), k=1)
        residuals = (
            cluster_anchors["value"].to_numpy(float)
            - cluster_prior["value"].to_numpy(float)[station_prior_index]
        )
        distances = np.hypot(
            output_x[:, None] - station_x[None, :],
            output_y[:, None] - station_y[None, :],
        )
        weights = np.exp(-0.5 * (distances / radius) ** 2)
        correction = _weighted_values(weights, residuals)
        correction *= np.exp(-0.5 * (distances.min(axis=1) / radius) ** 2)
        corrected = np.maximum(background + correction, 0.0)
        analysis.loc[indexes, "value"] = corrected
        analysis.loc[indexes, "analysis_value"] = corrected
        analysis.loc[indexes, "correction_value"] = correction
    return analysis


def _deposit_bilinear(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    x_min: float,
    y_min: float,
    step: float,
    shape: tuple[int, int],
) -> np.ndarray:
    output = np.zeros(shape, dtype=float)
    fx = (x - x_min) / step
    fy = (y - y_min) / step
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    for dx, dy, factor in (
        (0, 0, (1.0 - (fx - x0)) * (1.0 - (fy - y0))),
        (1, 0, (fx - x0) * (1.0 - (fy - y0))),
        (0, 1, (1.0 - (fx - x0)) * (fy - y0)),
        (1, 1, (fx - x0) * (fy - y0)),
    ):
        xi = x0 + dx
        yi = y0 + dy
        valid = (xi >= 0) & (xi < shape[1]) & (yi >= 0) & (yi < shape[0]) & np.isfinite(values)
        np.add.at(output, (yi[valid], xi[valid]), values[valid] * factor[valid])
    return output


def build_semiphysical_forecast_grid(
    analysis_grid: pd.DataFrame,
    forecast_anchors: pd.DataFrame,
    *,
    substance_code: str,
    wind_speed: float | None,
    wind_from_degrees: float | None,
    boundary_layer_height: float | None = None,
    precipitation: float | None = None,
) -> pd.DataFrame:
    """Advance the analysed field one hour and nudge it with CatBoost station corrections."""
    if analysis_grid.empty:
        return pd.DataFrame()
    settings = get_settings()
    step = float(settings.GRID_CELL_METERS)
    frames: list[pd.DataFrame] = []
    assigned_anchors = forecast_anchors.copy()
    if not assigned_anchors.empty and "cluster_id" not in assigned_anchors:
        centroids = analysis_grid.groupby("cluster_id")[["latitude", "longitude"]].mean()
        anchor_clusters: list[int] = []
        for anchor in assigned_anchors.itertuples(index=False):
            latitude_scale = METERS_PER_DEGREE_LATITUDE
            longitude_scale = latitude_scale * cos(float(anchor.latitude) * pi / 180.0)
            distances = np.hypot(
                (centroids["latitude"].to_numpy(float) - float(anchor.latitude)) * latitude_scale,
                (centroids["longitude"].to_numpy(float) - float(anchor.longitude)) * longitude_scale,
            )
            anchor_clusters.append(int(centroids.index[int(np.argmin(distances))]))
        assigned_anchors["cluster_id"] = anchor_clusters

    for cluster_id, source in analysis_grid.groupby("cluster_id", sort=False):
        source = source.reset_index(drop=True)
        reference_latitude = float(source["latitude"].mean())
        reference_longitude = float(source["longitude"].mean())
        source_x, source_y = _project(
            source["latitude"].to_numpy(float), source["longitude"].to_numpy(float),
            reference_latitude, reference_longitude,
        )
        anchors = assigned_anchors[assigned_anchors["cluster_id"] == cluster_id].copy()
        speed = max(float(wind_speed or 0.0), 0.0)
        direction = wind_from_degrees
        pbl = (
            float(boundary_layer_height)
            if boundary_layer_height is not None and np.isfinite(boundary_layer_height)
            else 500.0
        )
        rain = max(float(precipitation or 0.0), 0.0)
        if not anchors.empty:
            anchor_speeds = pd.to_numeric(
                anchors["weather_wind_speed"]
                if "weather_wind_speed" in anchors
                else pd.Series(np.nan, index=anchors.index),
                errors="coerce",
            )
            anchor_directions = pd.to_numeric(
                anchors["weather_wind_direction"]
                if "weather_wind_direction" in anchors
                else pd.Series(np.nan, index=anchors.index),
                errors="coerce",
            )
            usable_wind = anchor_speeds.notna() & anchor_directions.notna()
            if usable_wind.any():
                speed = max(float(anchor_speeds[usable_wind].mean()), 0.0)
                direction_radians = np.deg2rad(anchor_directions[usable_wind].to_numpy(float))
                direction = float((np.rad2deg(np.arctan2(
                    np.sin(direction_radians).mean(), np.cos(direction_radians).mean()
                )) + 360.0) % 360.0)
            anchor_pbl = pd.to_numeric(
                anchors["weather_boundary_layer_height"]
                if "weather_boundary_layer_height" in anchors
                else pd.Series(np.nan, index=anchors.index),
                errors="coerce",
            )
            if anchor_pbl.notna().any():
                pbl = float(anchor_pbl.mean())
            anchor_rain = pd.to_numeric(
                anchors["weather_precipitation"]
                if "weather_precipitation" in anchors
                else pd.Series(np.nan, index=anchors.index),
                errors="coerce",
            )
            if anchor_rain.notna().any():
                rain = max(float(anchor_rain.mean()), 0.0)
        wind_x, wind_y = _wind_components(speed, direction)
        displacement = min(
            speed * float(settings.PHYSICS_TIME_STEP_SECONDS),
            float(settings.PHYSICS_MAX_ADVECTION_METERS),
        )
        diffusivity = float(np.clip(
            settings.PHYSICS_MIN_DIFFUSIVITY_M2_S + 0.012 * pbl + 1.5 * speed,
            settings.PHYSICS_MIN_DIFFUSIVITY_M2_S,
            settings.PHYSICS_MAX_DIFFUSIVITY_M2_S,
        ))
        decay = GAS_DECAY_PER_HOUR.get(substance_code, 0.05) + GAS_RAIN_SCAVENGING.get(substance_code, 0.008) * rain
        attenuation = float(np.exp(-decay))
        shifted_x = source_x + wind_x * displacement
        shifted_y = source_y + wind_y * displacement
        anchor_x = np.asarray([], dtype=float)
        anchor_y = np.asarray([], dtype=float)
        if not anchors.empty:
            anchor_x, anchor_y = _project(
                anchors["latitude"].to_numpy(float), anchors["longitude"].to_numpy(float),
                reference_latitude, reference_longitude,
            )
        coverage_x = np.concatenate((
            shifted_x,
            anchor_x - float(settings.GRID_BUFFER_METERS),
            anchor_x + float(settings.GRID_BUFFER_METERS),
        ))
        coverage_y = np.concatenate((
            shifted_y,
            anchor_y - float(settings.GRID_BUFFER_METERS),
            anchor_y + float(settings.GRID_BUFFER_METERS),
        ))
        x_min = np.floor(coverage_x.min() / step) * step
        x_max = np.ceil(coverage_x.max() / step) * step
        y_min = np.floor(coverage_y.min() / step) * step
        y_max = np.ceil(coverage_y.max() / step) * step
        x_values = np.arange(x_min, x_max + step, step)
        y_values = np.arange(y_min, y_max + step, step)
        shape = (len(y_values), len(x_values))

        physical = _deposit_bilinear(
            shifted_x, shifted_y, source["value"].to_numpy(float), x_min, y_min, step, shape
        )
        transported_confidence = _deposit_bilinear(
            shifted_x, shifted_y, source["confidence"].to_numpy(float), x_min, y_min, step, shape
        )
        sigma_cells = max(
            np.sqrt(2.0 * diffusivity * float(settings.PHYSICS_TIME_STEP_SECONDS)) / step,
            0.35,
        )
        physical = gaussian_filter(physical, sigma=sigma_cells, mode="constant") * attenuation
        transported_confidence = np.clip(
            gaussian_filter(transported_confidence, sigma=sigma_cells, mode="constant"),
            0.0,
            1.0,
        )
        mesh_x, mesh_y = np.meshgrid(x_values, y_values)
        flat_x = mesh_x.ravel()
        flat_y = mesh_y.ravel()
        physical_flat = physical.ravel()
        correction = np.zeros_like(physical_flat)

        if not anchors.empty:
            physical_at_stations = physical[
                np.clip(np.rint((anchor_y - y_min) / step).astype(int), 0, shape[0] - 1),
                np.clip(np.rint((anchor_x - x_min) / step).astype(int), 0, shape[1] - 1),
            ]
            residuals = anchors["value"].to_numpy(float) - physical_at_stations
            distances = np.hypot(
                flat_x[:, None] - anchor_x[None, :],
                flat_y[:, None] - anchor_y[None, :],
            )
            radius = float(settings.PHYSICS_CORRECTION_RADIUS_METERS)
            weights = np.exp(-0.5 * (distances / radius) ** 2)
            correction = _weighted_values(weights, residuals)
            correction *= np.exp(-0.5 * (distances.min(axis=1) / radius) ** 2)
            anchor_confidence = np.exp(-0.5 * (distances.min(axis=1) / radius) ** 2)
            lower_bounds = _weighted_values(weights, anchors["lower_bound"].to_numpy(float))
            upper_bounds = _weighted_values(weights, anchors["upper_bound"].to_numpy(float))
        else:
            anchor_confidence = np.zeros_like(physical_flat)
            lower_bounds = np.full_like(physical_flat, np.nan)
            upper_bounds = np.full_like(physical_flat, np.nan)

        values = np.maximum(physical_flat + correction, 0.0)
        confidence = np.maximum(transported_confidence.ravel(), anchor_confidence)
        if anchor_x.size:
            station_distance = np.hypot(
                flat_x[:, None] - anchor_x[None, :],
                flat_y[:, None] - anchor_y[None, :],
            ).min(axis=1)
            transported_distance = np.hypot(
                flat_x[:, None] - (anchor_x + wind_x * displacement)[None, :],
                flat_y[:, None] - (anchor_y + wind_y * displacement)[None, :],
            ).min(axis=1)
            rounded_coverage = (
                (station_distance <= float(settings.GRID_BUFFER_METERS))
                | (transported_distance <= float(settings.GRID_BUFFER_METERS))
            )
        else:
            rounded_coverage = np.ones_like(values, dtype=bool)
        keep = np.isfinite(values) & (confidence >= 0.04) & rounded_coverage
        flat_x = flat_x[keep]
        flat_y = flat_y[keep]
        values = values[keep]
        physical_kept = physical_flat[keep]
        correction = correction[keep]
        confidence = confidence[keep]
        lower_bounds = np.where(np.isfinite(lower_bounds[keep]), np.minimum(lower_bounds[keep], values), np.nan)
        upper_bounds = np.where(np.isfinite(upper_bounds[keep]), np.maximum(upper_bounds[keep], values), np.nan)
        latitudes, longitudes = _unproject(flat_x, flat_y, reference_latitude, reference_longitude)
        south, west = _unproject(flat_x - step / 2, flat_y - step / 2, reference_latitude, reference_longitude)
        north, east = _unproject(flat_x + step / 2, flat_y + step / 2, reference_latitude, reference_longitude)
        frames.append(pd.DataFrame({
            "cluster_id": int(cluster_id),
            "grid_x": np.rint((flat_x - x_min) / step).astype(int),
            "grid_y": np.rint((flat_y - y_min) / step).astype(int),
            "latitude": latitudes,
            "longitude": longitudes,
            "south": south,
            "west": west,
            "north": north,
            "east": east,
            "value": values,
            "analysis_value": np.nan,
            "physical_forecast": physical_kept,
            "correction_value": correction,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
            "confidence": confidence,
            "source_station_count": int(source["source_station_count"].max()),
            "wind_speed": speed,
            "wind_direction": direction,
            "wind_u": wind_x * speed,
            "wind_v": wind_y * speed,
            "boundary_layer_height": pbl,
            "diffusion_coefficient": diffusivity,
            "decay_coefficient": decay,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
