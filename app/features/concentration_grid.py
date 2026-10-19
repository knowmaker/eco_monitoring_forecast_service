from __future__ import annotations

from math import cos, pi

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from app.config import get_settings
from app.features.physical_baseline import GAS_DECAY_PER_HOUR, GAS_RAIN_SCAVENGING


METERS_PER_DEGREE_LATITUDE = 111_320.0
SECONDS_PER_HOUR = 3600.0
COVARIANCE_AT_RADIUS = 0.05
MOSECOM_AGGREGATE_EFFECTIVE_SAMPLES = 3.0


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


def _wind_components(
    wind_speed: float | np.ndarray | None,
    wind_from_degrees: float | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.asarray(0.0 if wind_speed is None else wind_speed, dtype=float)
    direction = np.asarray(
        0.0 if wind_from_degrees is None else wind_from_degrees,
        dtype=float,
    )
    valid = np.isfinite(speed) & np.isfinite(direction)
    speed = np.where(valid, np.maximum(speed, 0.0), 0.0)
    wind_to_radians = np.deg2rad((np.where(valid, direction, 0.0) + 180.0) % 360.0)
    return np.sin(wind_to_radians) * speed, np.cos(wind_to_radians) * speed


def _wind_direction(wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    speed = np.hypot(wind_u, wind_v)
    wind_to = (np.rad2deg(np.arctan2(wind_u, wind_v)) + 360.0) % 360.0
    return np.where(speed > 0.0, (wind_to + 180.0) % 360.0, np.nan)


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _clean_anchors(anchors: pd.DataFrame) -> pd.DataFrame:
    required = {"monitoring_post_id", "latitude", "longitude", "value"}
    if anchors.empty or not required.issubset(anchors.columns):
        return pd.DataFrame()
    clean = anchors.copy()
    numeric_columns = (
        "latitude",
        "longitude",
        "value",
        "lower_bound",
        "upper_bound",
        "samples_count",
        "observation_confidence",
        "weather_wind_speed",
        "weather_wind_direction",
        "weather_boundary_layer_height",
        "weather_precipitation",
    )
    for column in numeric_columns:
        if column not in clean:
            clean[column] = np.nan
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["latitude", "longitude", "value"]).reset_index(drop=True)
    clean["value"] = clean["value"].clip(lower=0.0)
    clean["lower_bound"] = clean["lower_bound"].clip(lower=0.0)
    clean["upper_bound"] = clean["upper_bound"].clip(lower=0.0)
    return clean


def _observation_confidence(anchors: pd.DataFrame) -> np.ndarray:
    supplied = _numeric(anchors, "observation_confidence")
    confidence = np.where(np.isfinite(supplied), supplied, np.nan)

    samples = _numeric(anchors, "samples_count")
    if np.isfinite(samples).any():
        if "source" in anchors:
            samples = np.where(
                anchors["source"].eq("mosecom"),
                MOSECOM_AGGREGATE_EFFECTIVE_SAMPLES,
                samples,
            )
        maximum = max(float(np.nanmax(samples)), 1.0)
        sample_confidence = np.clip(samples / maximum, 0.05, 1.0)
        confidence = np.where(np.isfinite(confidence), confidence, sample_confidence)

    lower = _numeric(anchors, "lower_bound")
    upper = _numeric(anchors, "upper_bound")
    values = _numeric(anchors, "value")
    interval_width = upper - lower
    interval_confidence = 1.0 / (
        1.0 + np.maximum(interval_width, 0.0) / np.maximum(np.abs(values), 1e-9)
    )
    usable_interval = np.isfinite(interval_width) & (interval_width >= 0.0)
    confidence = np.where(
        np.isfinite(confidence),
        confidence,
        np.where(usable_interval, interval_confidence, 1.0),
    )
    return np.clip(confidence, 0.05, 1.0)


def _anchor_weather(
    anchors: pd.DataFrame,
    fallback_speed: float | None,
    fallback_direction: float | None,
    fallback_pbl: float | None = None,
    fallback_rain: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    speeds = _numeric(anchors, "weather_wind_speed")
    directions = _numeric(anchors, "weather_wind_direction")
    valid = np.isfinite(speeds) & np.isfinite(directions)
    default_speed = max(float(fallback_speed or 0.0), 0.0)
    default_direction = float(fallback_direction or 0.0)
    speeds = np.where(valid, np.maximum(speeds, 0.0), default_speed)
    directions = np.where(valid, directions, default_direction)
    wind_u, wind_v = _wind_components(speeds, directions)

    pbl = _numeric(anchors, "weather_boundary_layer_height")
    pbl_default = (
        float(fallback_pbl)
        if fallback_pbl is not None and np.isfinite(fallback_pbl)
        else 500.0
    )
    pbl = np.where(np.isfinite(pbl), np.clip(pbl, 50.0, 4000.0), pbl_default)
    rain = _numeric(anchors, "weather_precipitation")
    rain_default = max(float(fallback_rain or 0.0), 0.0)
    rain = np.where(np.isfinite(rain), np.maximum(rain, 0.0), rain_default)
    return wind_u, wind_v, pbl, rain


def _interpolate_nearest_values(
    query_x: np.ndarray,
    query_y: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    values: np.ndarray,
    fallback: float,
) -> np.ndarray:
    finite = np.isfinite(source_x) & np.isfinite(source_y) & np.isfinite(values)
    if not finite.any():
        return np.full(len(query_x), fallback, dtype=float)
    points = np.column_stack((source_x[finite], source_y[finite]))
    usable_values = values[finite]
    count = min(4, len(points))
    distances, indexes = cKDTree(points).query(
        np.column_stack((query_x, query_y)),
        k=count,
    )
    if count == 1:
        return usable_values[np.asarray(indexes, dtype=int)]
    distances = np.asarray(distances, dtype=float)
    indexes = np.asarray(indexes, dtype=int)
    exact = distances[:, 0] <= 1e-9
    weights = 1.0 / np.maximum(distances, 1.0) ** 2
    interpolated = np.sum(weights * usable_values[indexes], axis=1) / np.sum(weights, axis=1)
    interpolated[exact] = usable_values[indexes[exact, 0]]
    return interpolated


def _anisotropic_correlation(
    target_x: np.ndarray,
    target_y: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_wind_u: np.ndarray,
    source_wind_v: np.ndarray,
    radius: float,
) -> np.ndarray:
    delta_x = target_x[:, None] - source_x[None, :]
    delta_y = target_y[:, None] - source_y[None, :]
    distances = np.hypot(delta_x, delta_y)
    speed = np.hypot(source_wind_u, source_wind_v)
    unit_u = np.divide(source_wind_u, speed, out=np.zeros_like(speed), where=speed > 0.1)
    unit_v = np.divide(source_wind_v, speed, out=np.ones_like(speed), where=speed > 0.1)
    along = delta_x * unit_u[None, :] + delta_y * unit_v[None, :]
    across = -delta_x * unit_v[None, :] + delta_y * unit_u[None, :]

    sigma_along = radius / np.sqrt(-2.0 * np.log(COVARIANCE_AT_RADIUS))
    anisotropy = 1.0 + np.minimum(speed / 5.0, 2.0)
    sigma_across = sigma_along / anisotropy
    sigma_across = np.where(speed <= 0.1, sigma_along, sigma_across)
    normalized = (along / sigma_along) ** 2 + (across / sigma_across[None, :]) ** 2
    correlation = np.exp(-0.5 * normalized)
    correlation[distances > radius] = 0.0
    return correlation


def _sample_grid(
    query_x: np.ndarray,
    query_y: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    values: np.ndarray,
    step: float,
    default: float,
) -> np.ndarray:
    if not len(grid_x):
        return np.full(len(query_x), default, dtype=float)
    distances, indexes = cKDTree(np.column_stack((grid_x, grid_y))).query(
        np.column_stack((query_x, query_y)),
        k=1,
    )
    sampled = np.full(len(query_x), default, dtype=float)
    usable = (distances <= step * 0.8) & np.isfinite(values[indexes])
    sampled[usable] = values[indexes[usable]]
    return sampled


def _optimal_interpolation(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    background: np.ndarray,
    background_confidence: np.ndarray,
    observation_x: np.ndarray,
    observation_y: np.ndarray,
    observation_values: np.ndarray,
    observation_confidence: np.ndarray,
    observation_wind_u: np.ndarray,
    observation_wind_v: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(observation_x):
        return background.copy(), background_confidence.copy(), np.zeros_like(background)

    radius = float(get_settings().MAX_OBSERVATION_INFLUENCE_METERS)
    background_at_observations = _sample_grid(
        observation_x,
        observation_y,
        grid_x,
        grid_y,
        background,
        step,
        0.0,
    )
    confidence_at_observations = _sample_grid(
        observation_x,
        observation_y,
        grid_x,
        grid_y,
        background_confidence,
        step,
        0.0,
    )
    innovations = observation_values - background_at_observations
    grid_variance = np.clip(1.0 - background_confidence, 0.05, 1.0)
    observation_background_variance = np.clip(
        1.0 - confidence_at_observations,
        0.05,
        1.0,
    )

    correlation_oo = _anisotropic_correlation(
        observation_x,
        observation_y,
        observation_x,
        observation_y,
        observation_wind_u,
        observation_wind_v,
        radius,
    )
    correlation_oo = (correlation_oo + correlation_oo.T) / 2.0
    np.fill_diagonal(correlation_oo, 1.0)
    covariance_oo = correlation_oo * np.sqrt(
        observation_background_variance[:, None]
        * observation_background_variance[None, :]
    )
    observation_error = np.clip(1.0 - observation_confidence, 0.01, 1.0)
    system = covariance_oo + np.diag(observation_error)

    covariance_go = _anisotropic_correlation(
        grid_x,
        grid_y,
        observation_x,
        observation_y,
        observation_wind_u,
        observation_wind_v,
        radius,
    ) * np.sqrt(grid_variance[:, None] * observation_background_variance[None, :])
    try:
        coefficients = np.linalg.solve(system, innovations)
        solved_covariance = np.linalg.solve(system, covariance_go.T).T
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, innovations, rcond=None)[0]
        solved_covariance = np.linalg.lstsq(system, covariance_go.T, rcond=None)[0].T

    increment = covariance_go @ coefficients
    variance_reduction = np.sum(covariance_go * solved_covariance, axis=1)
    posterior_variance = np.clip(grid_variance - variance_reduction, 0.0, 1.0)
    confidence = np.maximum(background_confidence, 1.0 - posterior_variance)
    return np.maximum(background + increment, 0.0), np.clip(confidence, 0.0, 1.0), increment


def _backward_innovation(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    observation_x: np.ndarray,
    observation_y: np.ndarray,
    innovations: np.ndarray,
    observation_confidence: np.ndarray,
    observation_wind_u: np.ndarray,
    observation_wind_v: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    correction_numerator = np.zeros(len(grid_x), dtype=float)
    correction_weight = np.zeros(len(grid_x), dtype=float)
    radius = float(get_settings().MAX_OBSERVATION_INFLUENCE_METERS)

    for index in range(len(observation_x)):
        path_x = -observation_wind_u[index] * SECONDS_PER_HOUR
        path_y = -observation_wind_v[index] * SECONDS_PER_HOUR
        displacement = np.hypot(path_x, path_y)
        if displacement < step or not np.isfinite(innovations[index]):
            continue
        relative_x = grid_x - observation_x[index]
        relative_y = grid_y - observation_y[index]
        fraction = np.clip(
            (relative_x * path_x + relative_y * path_y) / displacement**2,
            step / displacement,
            1.0,
        )
        closest_x = observation_x[index] + path_x * fraction
        closest_y = observation_y[index] + path_y * fraction
        delta_x = grid_x - closest_x
        delta_y = grid_y - closest_y
        distance = np.hypot(delta_x, delta_y)
        speed = np.hypot(observation_wind_u[index], observation_wind_v[index])
        unit_u = observation_wind_u[index] / speed
        unit_v = observation_wind_v[index] / speed
        along = delta_x * unit_u + delta_y * unit_v
        across = -delta_x * unit_v + delta_y * unit_u
        sigma_along = radius / np.sqrt(-2.0 * np.log(COVARIANCE_AT_RADIUS))
        sigma_across = sigma_along / (1.0 + min(speed / 5.0, 2.0))
        strongest = np.exp(
            -0.5 * ((along / sigma_along) ** 2 + (across / sigma_across) ** 2)
        )
        strongest[distance > radius] = 0.0
        strongest *= np.exp(-fraction) * observation_confidence[index]
        correction_numerator += strongest * innovations[index]
        correction_weight += strongest

    correction = np.divide(
        correction_numerator,
        correction_weight,
        out=np.zeros_like(correction_numerator),
        where=correction_weight > 0.0,
    )
    return correction, np.clip(correction_weight, 0.0, 1.0)


def _frame_geometry(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    x_min: float,
    y_min: float,
    step: float,
    reference_latitude: float,
    reference_longitude: float,
) -> dict[str, np.ndarray]:
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
    return {
        "grid_x": np.rint((grid_x - x_min) / step).astype(int),
        "grid_y": np.rint((grid_y - y_min) / step).astype(int),
        "latitude": latitudes,
        "longitude": longitudes,
        "south": south,
        "west": west,
        "north": north,
        "east": east,
    }


def _grid_extent(
    support_x: np.ndarray,
    support_y: np.ndarray,
    step: float,
    padding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x_min = np.floor((support_x.min() - padding) / step) * step
    x_max = np.ceil((support_x.max() + padding) / step) * step
    y_min = np.floor((support_y.min() - padding) / step) * step
    y_max = np.ceil((support_y.max() + padding) / step) * step
    x_values = np.arange(x_min, x_max + step, step)
    y_values = np.arange(y_min, y_max + step, step)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    return x_values, y_values, mesh_x.ravel(), mesh_y.ravel(), x_min, y_min


def assimilate_observations(
    anchors: pd.DataFrame,
    prior_grid: pd.DataFrame | None,
    *,
    wind_speed: float | None = None,
    wind_from_degrees: float | None = None,
) -> pd.DataFrame:
    """Combine observations with the previous forecast using flow-dependent covariance."""
    clean = _clean_anchors(anchors)
    if clean.empty:
        return pd.DataFrame()

    prior = prior_grid.copy() if prior_grid is not None else pd.DataFrame()
    latitude_values = [clean["latitude"]]
    longitude_values = [clean["longitude"]]
    if not prior.empty:
        latitude_values.append(prior["latitude"])
        longitude_values.append(prior["longitude"])
    reference_latitude = float(pd.concat(latitude_values, ignore_index=True).mean())
    reference_longitude = float(pd.concat(longitude_values, ignore_index=True).mean())
    observation_x, observation_y = _project(
        clean["latitude"].to_numpy(float),
        clean["longitude"].to_numpy(float),
        reference_latitude,
        reference_longitude,
    )
    observation_u, observation_v, _, _ = _anchor_weather(
        clean,
        wind_speed,
        wind_from_degrees,
    )

    support_x = observation_x.copy()
    support_y = observation_y.copy()
    prior_x = np.asarray([], dtype=float)
    prior_y = np.asarray([], dtype=float)
    if not prior.empty:
        prior_x, prior_y = _project(
            prior["latitude"].to_numpy(float),
            prior["longitude"].to_numpy(float),
            reference_latitude,
            reference_longitude,
        )
        support_x = np.concatenate((support_x, prior_x, observation_x - observation_u * SECONDS_PER_HOUR))
        support_y = np.concatenate((support_y, prior_y, observation_y - observation_v * SECONDS_PER_HOUR))

    settings = get_settings()
    step = float(settings.GRID_CELL_METERS)
    radius = float(settings.MAX_OBSERVATION_INFLUENCE_METERS)
    _, _, grid_x, grid_y, x_min, y_min = _grid_extent(
        support_x,
        support_y,
        step,
        radius,
    )

    if prior.empty:
        background = np.zeros(len(grid_x), dtype=float)
        background_confidence = np.zeros(len(grid_x), dtype=float)
        lower_bounds = np.full(len(grid_x), np.nan)
        upper_bounds = np.full(len(grid_x), np.nan)
        prior_coverage = np.zeros(len(grid_x), dtype=bool)
    else:
        prior_values = _numeric(prior, "value", 0.0)
        prior_confidence = _numeric(prior, "confidence", 0.0)
        prior_tree = cKDTree(np.column_stack((prior_x, prior_y)))
        prior_distance, prior_index = prior_tree.query(np.column_stack((grid_x, grid_y)), k=1)
        prior_coverage = prior_distance <= step * 0.8
        background = np.where(prior_coverage, prior_values[prior_index], 0.0)
        background_confidence = np.where(
            prior_coverage,
            np.nan_to_num(prior_confidence[prior_index], nan=0.0),
            0.0,
        )
        prior_lower = _numeric(prior, "lower_bound")
        prior_upper = _numeric(prior, "upper_bound")
        lower_bounds = np.where(prior_coverage, prior_lower[prior_index], np.nan)
        upper_bounds = np.where(prior_coverage, prior_upper[prior_index], np.nan)

    observation_confidence = _observation_confidence(clean)
    values, confidence, direct_increment = _optimal_interpolation(
        grid_x,
        grid_y,
        background,
        background_confidence,
        observation_x,
        observation_y,
        clean["value"].to_numpy(float),
        observation_confidence,
        observation_u,
        observation_v,
        step,
    )

    backward_increment = np.zeros(len(grid_x), dtype=float)
    backward_confidence = np.zeros(len(grid_x), dtype=float)
    if not prior.empty:
        prior_at_observations = _sample_grid(
            observation_x,
            observation_y,
            prior_x,
            prior_y,
            _numeric(prior, "value", 0.0),
            step,
            0.0,
        )
        innovations = clean["value"].to_numpy(float) - prior_at_observations
        backward_increment, backward_confidence = _backward_innovation(
            grid_x,
            grid_y,
            observation_x,
            observation_y,
            innovations,
            observation_confidence,
            observation_u,
            observation_v,
            step,
        )
        values = np.maximum(values + backward_increment * (1.0 - confidence), 0.0)
        confidence = np.maximum(confidence, backward_confidence)

    station_distance = cKDTree(np.column_stack((observation_x, observation_y))).query(
        np.column_stack((grid_x, grid_y)),
        k=1,
    )[0]
    observation_coverage = station_distance <= radius
    backward_coverage = backward_confidence >= settings.GRID_MIN_CONFIDENCE
    keep = (
        np.isfinite(values)
        & (confidence >= settings.GRID_MIN_CONFIDENCE)
        & (prior_coverage | observation_coverage | backward_coverage)
    )

    grid_x = grid_x[keep]
    grid_y = grid_y[keep]
    values = values[keep]
    confidence = confidence[keep]
    correction = (direct_increment + backward_increment)[keep]
    lower_bounds = np.where(
        np.isfinite(lower_bounds[keep]),
        np.minimum(lower_bounds[keep], values),
        np.nan,
    )
    upper_bounds = np.where(
        np.isfinite(upper_bounds[keep]),
        np.maximum(upper_bounds[keep], values),
        np.nan,
    )

    local_u = _interpolate_nearest_values(
        grid_x, grid_y, observation_x, observation_y, observation_u, float(np.mean(observation_u))
    )
    local_v = _interpolate_nearest_values(
        grid_x, grid_y, observation_x, observation_y, observation_v, float(np.mean(observation_v))
    )
    geometry = _frame_geometry(
        grid_x,
        grid_y,
        x_min,
        y_min,
        step,
        reference_latitude,
        reference_longitude,
    )
    speed = np.hypot(local_u, local_v)
    return pd.DataFrame({
        **geometry,
        "value": values,
        "analysis_value": values,
        "physical_forecast": np.nan,
        "correction_value": correction,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "confidence": confidence,
        "source_station_count": clean["monitoring_post_id"].nunique(),
        "wind_speed": speed,
        "wind_direction": _wind_direction(local_u, local_v),
        "wind_u": local_u,
        "wind_v": local_v,
        "boundary_layer_height": np.nan,
        "diffusion_coefficient": np.nan,
        "decay_coefficient": np.nan,
    })


def _deposit_average(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    x_min: float,
    y_min: float,
    step: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.zeros(shape, dtype=float)
    denominator = np.zeros(shape, dtype=float)
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
        valid = (
            (xi >= 0)
            & (xi < shape[1])
            & (yi >= 0)
            & (yi < shape[0])
            & np.isfinite(values)
        )
        np.add.at(numerator, (yi[valid], xi[valid]), values[valid] * factor[valid])
        np.add.at(denominator, (yi[valid], xi[valid]), factor[valid])
    averaged = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return averaged, denominator


def build_semiphysical_forecast_grid(
    analysis_grid: pd.DataFrame,
    forecast_anchors: pd.DataFrame,
    *,
    substance_code: str,
    wind_speed: float | None,
    wind_from_degrees: float | None,
    weather_anchors: pd.DataFrame | None = None,
    boundary_layer_height: float | None = None,
    precipitation: float | None = None,
    duration_seconds: float = SECONDS_PER_HOUR,
) -> pd.DataFrame:
    """Transport the complete field and assimilate CatBoost station corrections."""
    if analysis_grid.empty:
        return pd.DataFrame()
    anchors = _clean_anchors(forecast_anchors)
    weather_sources = _clean_anchors(weather_anchors) if weather_anchors is not None else anchors
    settings = get_settings()
    step = float(settings.GRID_CELL_METERS)
    radius = float(settings.MAX_OBSERVATION_INFLUENCE_METERS)
    duration_seconds = max(float(duration_seconds), 1.0)
    duration_hours = duration_seconds / SECONDS_PER_HOUR

    latitude_values = [analysis_grid["latitude"]]
    longitude_values = [analysis_grid["longitude"]]
    if not anchors.empty:
        latitude_values.append(anchors["latitude"])
        longitude_values.append(anchors["longitude"])
    if weather_anchors is not None and not weather_sources.empty:
        latitude_values.append(weather_sources["latitude"])
        longitude_values.append(weather_sources["longitude"])
    reference_latitude = float(pd.concat(latitude_values, ignore_index=True).mean())
    reference_longitude = float(pd.concat(longitude_values, ignore_index=True).mean())
    source_x, source_y = _project(
        analysis_grid["latitude"].to_numpy(float),
        analysis_grid["longitude"].to_numpy(float),
        reference_latitude,
        reference_longitude,
    )

    anchor_x = np.asarray([], dtype=float)
    anchor_y = np.asarray([], dtype=float)
    fallback_u, fallback_v = _wind_components(wind_speed, wind_from_degrees)
    fallback_u_value = float(np.asarray(fallback_u))
    fallback_v_value = float(np.asarray(fallback_v))
    if anchors.empty:
        anchor_x = np.asarray([], dtype=float)
        anchor_y = np.asarray([], dtype=float)
    else:
        anchor_x, anchor_y = _project(
            anchors["latitude"].to_numpy(float),
            anchors["longitude"].to_numpy(float),
            reference_latitude,
            reference_longitude,
        )
    if weather_sources.empty:
        weather_x = np.asarray([], dtype=float)
        weather_y = np.asarray([], dtype=float)
        weather_u = np.asarray([], dtype=float)
        weather_v = np.asarray([], dtype=float)
        weather_pbl = np.asarray([], dtype=float)
        weather_rain = np.asarray([], dtype=float)
    else:
        weather_x, weather_y = _project(
            weather_sources["latitude"].to_numpy(float),
            weather_sources["longitude"].to_numpy(float),
            reference_latitude,
            reference_longitude,
        )
        weather_u, weather_v, weather_pbl, weather_rain = _anchor_weather(
            weather_sources,
            wind_speed,
            wind_from_degrees,
            boundary_layer_height,
            precipitation,
        )

    maximum_speed = max(
        float(np.hypot(fallback_u_value, fallback_v_value)),
        float(np.max(np.hypot(weather_u, weather_v))) if len(weather_u) else 0.0,
    )
    substeps = max(1, int(np.ceil(maximum_speed * duration_seconds / step)))
    time_step = duration_seconds / substeps
    transported_x = source_x.copy()
    transported_y = source_y.copy()
    for _ in range(substeps):
        local_u = _interpolate_nearest_values(
            transported_x,
            transported_y,
            weather_x,
            weather_y,
            weather_u,
            fallback_u_value,
        )
        local_v = _interpolate_nearest_values(
            transported_x,
            transported_y,
            weather_x,
            weather_y,
            weather_v,
            fallback_v_value,
        )
        transported_x += local_u * time_step
        transported_y += local_v * time_step

    support_x = np.concatenate((transported_x, anchor_x)) if len(anchor_x) else transported_x
    support_y = np.concatenate((transported_y, anchor_y)) if len(anchor_y) else transported_y
    padding = radius if len(anchor_x) else step
    x_values, y_values, grid_x, grid_y, x_min, y_min = _grid_extent(
        support_x,
        support_y,
        step,
        padding,
    )
    shape = (len(y_values), len(x_values))
    physical, _ = _deposit_average(
        transported_x,
        transported_y,
        _numeric(analysis_grid, "value", 0.0),
        x_min,
        y_min,
        step,
        shape,
    )
    transported_confidence, _ = _deposit_average(
        transported_x,
        transported_y,
        _numeric(analysis_grid, "confidence", 0.0),
        x_min,
        y_min,
        step,
        shape,
    )

    grid_u = _interpolate_nearest_values(
        grid_x, grid_y, weather_x, weather_y, weather_u, fallback_u_value
    )
    grid_v = _interpolate_nearest_values(
        grid_x, grid_y, weather_x, weather_y, weather_v, fallback_v_value
    )
    pbl_default = (
        float(boundary_layer_height)
        if boundary_layer_height is not None and np.isfinite(boundary_layer_height)
        else 500.0
    )
    rain_default = max(float(precipitation or 0.0), 0.0)
    grid_pbl = _interpolate_nearest_values(
        grid_x, grid_y, weather_x, weather_y, weather_pbl, pbl_default
    )
    grid_rain = _interpolate_nearest_values(
        grid_x, grid_y, weather_x, weather_y, weather_rain, rain_default
    )
    grid_speed = np.hypot(grid_u, grid_v)
    diffusivity = np.clip(
        settings.PHYSICS_MIN_DIFFUSIVITY_M2_S + 0.012 * grid_pbl + 1.5 * grid_speed,
        settings.PHYSICS_MIN_DIFFUSIVITY_M2_S,
        settings.PHYSICS_MAX_DIFFUSIVITY_M2_S,
    )
    sigma_cells = max(
        np.sqrt(2.0 * float(np.mean(diffusivity)) * duration_seconds) / step,
        0.35,
    )
    physical = gaussian_filter(physical, sigma=sigma_cells, mode="constant").ravel()
    transported_confidence = np.clip(
        gaussian_filter(transported_confidence, sigma=sigma_cells, mode="constant").ravel()
        * settings.PHYSICS_CONFIDENCE_RETENTION_PER_HOUR**duration_hours,
        0.0,
        1.0,
    )
    decay = (
        GAS_DECAY_PER_HOUR.get(substance_code, 0.05)
        + GAS_RAIN_SCAVENGING.get(substance_code, 0.008) * grid_rain
    )
    physical *= np.exp(-decay * duration_hours)

    if anchors.empty:
        values = physical.copy()
        confidence = transported_confidence.copy()
        correction = np.zeros_like(values)
        lower_bounds = np.full_like(values, np.nan)
        upper_bounds = np.full_like(values, np.nan)
        station_coverage = np.zeros(len(values), dtype=bool)
    else:
        anchor_confidence = _observation_confidence(anchors)
        values, confidence, correction = _optimal_interpolation(
            grid_x,
            grid_y,
            physical,
            transported_confidence,
            anchor_x,
            anchor_y,
            anchors["value"].to_numpy(float),
            anchor_confidence,
            weather_u,
            weather_v,
            step,
        )
        lower_values = anchors["lower_bound"].to_numpy(float)
        upper_values = anchors["upper_bound"].to_numpy(float)
        if np.isfinite(lower_values).any():
            usable = np.isfinite(lower_values)
            lower_bounds = _optimal_interpolation(
                grid_x, grid_y, physical, transported_confidence,
                anchor_x[usable], anchor_y[usable], lower_values[usable],
                anchor_confidence[usable], weather_u[usable], weather_v[usable], step,
            )[0]
        else:
            lower_bounds = np.full_like(values, np.nan)
        if np.isfinite(upper_values).any():
            usable = np.isfinite(upper_values)
            upper_bounds = _optimal_interpolation(
                grid_x, grid_y, physical, transported_confidence,
                anchor_x[usable], anchor_y[usable], upper_values[usable],
                anchor_confidence[usable], weather_u[usable], weather_v[usable], step,
            )[0]
        else:
            upper_bounds = np.full_like(values, np.nan)
        station_distance = cKDTree(np.column_stack((anchor_x, anchor_y))).query(
            np.column_stack((grid_x, grid_y)),
            k=1,
        )[0]
        station_coverage = station_distance <= radius

    transported_coverage = transported_confidence >= settings.GRID_MIN_CONFIDENCE
    significant = values >= settings.PHYSICS_MIN_CONCENTRATION
    keep = (
        np.isfinite(values)
        & (confidence >= settings.GRID_MIN_CONFIDENCE)
        & (station_coverage | (transported_coverage & (significant | anchors.empty)))
    )
    grid_x = grid_x[keep]
    grid_y = grid_y[keep]
    values = np.maximum(values[keep], 0.0)
    physical_kept = np.maximum(physical[keep], 0.0)
    correction = correction[keep]
    confidence = confidence[keep]
    lower_bounds = np.where(
        np.isfinite(lower_bounds[keep]),
        np.minimum(lower_bounds[keep], values),
        np.nan,
    )
    upper_bounds = np.where(
        np.isfinite(upper_bounds[keep]),
        np.maximum(upper_bounds[keep], values),
        np.nan,
    )
    geometry = _frame_geometry(
        grid_x,
        grid_y,
        x_min,
        y_min,
        step,
        reference_latitude,
        reference_longitude,
    )
    source_station_count = (
        anchors["monitoring_post_id"].nunique()
        if not anchors.empty
        else int(np.nanmax(_numeric(analysis_grid, "source_station_count", 0.0)))
    )
    return pd.DataFrame({
        **geometry,
        "value": values,
        "analysis_value": np.nan,
        "physical_forecast": physical_kept,
        "correction_value": correction,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "confidence": confidence,
        "source_station_count": source_station_count,
        "wind_speed": grid_speed[keep],
        "wind_direction": _wind_direction(grid_u[keep], grid_v[keep]),
        "wind_u": grid_u[keep],
        "wind_v": grid_v[keep],
        "boundary_layer_height": grid_pbl[keep],
        "diffusion_coefficient": diffusivity[keep],
        "decay_coefficient": decay[keep],
    })
