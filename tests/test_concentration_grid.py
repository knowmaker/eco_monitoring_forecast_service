from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.features.concentration_grid import (
    assimilate_observations,
    build_semiphysical_forecast_grid,
)
from app.inference.concentration_grid import _attach_live_weather
from app.inference import concentration_grid as stored_grids


def _anchors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "monitoring_post_id": 1,
                "latitude": 55.9505,
                "longitude": 38.1246,
                "value": 0.04,
                "lower_bound": 0.03,
                "upper_bound": 0.05,
            },
            {
                "monitoring_post_id": 2,
                "latitude": 55.9508,
                "longitude": 38.1265,
                "value": 0.08,
                "lower_bound": 0.06,
                "upper_bound": 0.10,
            },
            {
                "monitoring_post_id": 3,
                "latitude": 55.9518,
                "longitude": 38.1251,
                "value": 0.12,
                "lower_bound": 0.09,
                "upper_bound": 0.15,
            },
            {
                "monitoring_post_id": 4,
                "latitude": 55.7669,
                "longitude": 37.6805,
                "value": 0.50,
                "lower_bound": 0.40,
                "upper_bound": 0.60,
            },
        ]
    )


def _offset(latitude: float, longitude: float, east_m: float, north_m: float) -> tuple[float, float]:
    return (
        latitude + north_m / 111_320.0,
        longitude + east_m / (111_320.0 * np.cos(np.deg2rad(latitude))),
    )


def _nearest_row(frame: pd.DataFrame, latitude: float, longitude: float) -> pd.Series:
    north = (frame["latitude"] - latitude) * 111_320.0
    east = (
        (frame["longitude"] - longitude)
        * 111_320.0
        * np.cos(np.deg2rad(latitude))
    )
    return frame.iloc[int(np.argmin(np.hypot(east, north)))]


def test_analysis_uses_one_field_without_cluster_identifiers():
    analysis = assimilate_observations(
        _anchors(),
        None,
        wind_speed=2.0,
        wind_from_degrees=270.0,
    )

    assert not analysis.empty
    assert "cluster_id" not in analysis
    assert set(analysis["source_station_count"]) == {4}
    assert analysis["confidence"].between(0.0, 1.0).all()
    assert (analysis["south"] < analysis["north"]).all()
    assert (analysis["west"] < analysis["east"]).all()


def test_single_station_influence_never_exceeds_ten_kilometers():
    anchor = _anchors().iloc[:1]
    analysis = assimilate_observations(anchor, None)
    latitude = float(anchor.iloc[0].latitude)
    longitude = float(anchor.iloc[0].longitude)
    north = (analysis["latitude"] - latitude) * 111_320.0
    east = (
        (analysis["longitude"] - longitude)
        * 111_320.0
        * np.cos(np.deg2rad(latitude))
    )
    distances = np.hypot(east, north)

    assert distances.max() <= 10_000.01
    assert distances.max() > 5_000.0


def test_observation_influence_is_longer_along_wind_than_across_it():
    anchor = _anchors().iloc[:1].copy()
    anchor["weather_wind_speed"] = 5.0
    anchor["weather_wind_direction"] = 270.0
    analysis = assimilate_observations(anchor, None)
    latitude = float(anchor.iloc[0].latitude)
    longitude = float(anchor.iloc[0].longitude)
    north = np.abs((analysis["latitude"] - latitude) * 111_320.0)
    east = np.abs(
        (analysis["longitude"] - longitude)
        * 111_320.0
        * np.cos(np.deg2rad(latitude))
    )

    assert east.max() > north.max()


def test_stations_with_overlapping_influence_have_a_field_between_them():
    latitude = 55.75
    longitude = 37.6
    second_latitude, second_longitude = _offset(latitude, longitude, 15_000.0, 0.0)
    anchors = pd.DataFrame([
        {"monitoring_post_id": 1, "latitude": latitude, "longitude": longitude, "value": 0.02},
        {
            "monitoring_post_id": 2,
            "latitude": second_latitude,
            "longitude": second_longitude,
            "value": 0.10,
        },
    ])

    analysis = assimilate_observations(anchors, None)
    middle_latitude, middle_longitude = _offset(latitude, longitude, 7_500.0, 0.0)
    middle = _nearest_row(analysis, middle_latitude, middle_longitude)

    assert 0.02 < middle["value"] < 0.10
    assert middle["confidence"] > 0.0


def test_stations_farther_than_two_influence_radii_leave_a_gap():
    latitude = 55.75
    longitude = 37.6
    second_latitude, second_longitude = _offset(latitude, longitude, 25_000.0, 0.0)
    anchors = pd.DataFrame([
        {"monitoring_post_id": 1, "latitude": latitude, "longitude": longitude, "value": 0.02},
        {
            "monitoring_post_id": 2,
            "latitude": second_latitude,
            "longitude": second_longitude,
            "value": 0.10,
        },
    ])

    analysis = assimilate_observations(anchors, None)
    middle_latitude, middle_longitude = _offset(latitude, longitude, 12_500.0, 0.0)
    middle = _nearest_row(analysis, middle_latitude, middle_longitude)
    north = (middle["latitude"] - middle_latitude) * 111_320.0
    east = (
        (middle["longitude"] - middle_longitude)
        * 111_320.0
        * np.cos(np.deg2rad(middle_latitude))
    )

    assert np.hypot(east, north) > 1_000.0


def test_negative_observation_does_not_create_concentration_mass():
    anchors = _anchors().iloc[:1].copy()
    anchors["value"] = -0.12

    analysis = assimilate_observations(anchors, None)

    assert (analysis["value"] >= 0.0).all()
    assert analysis["value"].max() == 0.0


def test_forecast_moves_field_by_wind_without_fixed_distance_cap():
    anchors = _anchors().iloc[:1].copy()
    analysis = assimilate_observations(anchors, None)
    forecast_anchors = anchors.copy()
    forecast_anchors["weather_wind_speed"] = 4.0
    forecast_anchors["weather_wind_direction"] = 270.0
    forecast = build_semiphysical_forecast_grid(
        analysis,
        forecast_anchors,
        substance_code="NO2",
        wind_speed=4.0,
        wind_from_degrees=270.0,
    )

    assert forecast["east"].max() > analysis["east"].max() + 0.08


def test_forecast_uses_spatially_varying_wind_vectors():
    anchors = _anchors().iloc[:2].copy()
    analysis = assimilate_observations(anchors, None)
    forecast_anchors = anchors.copy()
    forecast_anchors["weather_wind_speed"] = [3.0, 3.0]
    forecast_anchors["weather_wind_direction"] = [270.0, 90.0]
    forecast = build_semiphysical_forecast_grid(
        analysis,
        forecast_anchors,
        substance_code="NO2",
        wind_speed=3.0,
        wind_from_degrees=270.0,
    )

    assert forecast["wind_u"].min() < 0.0
    assert forecast["wind_u"].max() > 0.0


def test_nowcast_uses_local_weather_without_station_predictions():
    anchors = _anchors().iloc[:2].copy()
    analysis = assimilate_observations(anchors, None)
    weather = anchors.copy()
    weather["weather_wind_speed"] = [3.0, 3.0]
    weather["weather_wind_direction"] = [270.0, 90.0]

    nowcast = build_semiphysical_forecast_grid(
        analysis,
        pd.DataFrame(),
        substance_code="NO2",
        wind_speed=3.0,
        wind_from_degrees=270.0,
        weather_anchors=weather,
    )

    assert not nowcast.empty
    assert nowcast["wind_u"].min() < 0.0
    assert nowcast["wind_u"].max() > 0.0
    assert (nowcast["correction_value"] == 0.0).all()
    assert set(nowcast["source_station_count"]) == {2}


def test_nowcast_uses_historical_weather_when_live_station_weather_is_missing():
    sources = _anchors().iloc[:2].copy()
    sources["weather_wind_speed"] = [2.0, 3.0]
    live = pd.DataFrame([{
        "monitoring_post_id": 1,
        "wind_speed_100m": 5.0,
        "wind_direction_100m": 270.0,
        "boundary_layer_height": 600.0,
        "precipitation": 0.0,
    }])

    weather_sources = _attach_live_weather(sources, live)

    assert weather_sources["weather_wind_speed"].tolist() == [5.0, 3.0]


def test_zero_concentration_nowcast_remains_available():
    anchor = _anchors().iloc[:1].copy()
    anchor["value"] = 0.0
    analysis = assimilate_observations(anchor, None)

    nowcast = build_semiphysical_forecast_grid(
        analysis,
        pd.DataFrame(),
        substance_code="NO2",
        wind_speed=0.0,
        wind_from_degrees=0.0,
        weather_anchors=anchor,
    )

    assert not nowcast.empty
    assert (nowcast["value"] == 0.0).all()


def test_missing_analysis_keeps_existing_current_hour_forecast(monkeypatch):
    def reject_delete(*_args):
        raise AssertionError("forecast was deleted")

    monkeypatch.setattr(stored_grids, "_stored_grid", lambda *_: pd.DataFrame())
    monkeypatch.setattr(stored_grids, "_replace_grid", reject_delete)

    result = stored_grids.build_nowcast_grid(
        None, "NO2", datetime(2026, 9, 25, 10, tzinfo=timezone.utc)
    )

    assert result == 0


def test_next_hour_forecast_uses_nowcast_as_background():
    anchors = _anchors().iloc[:2].copy()
    analysis = assimilate_observations(anchors, None)
    nowcast = build_semiphysical_forecast_grid(
        analysis,
        pd.DataFrame(),
        substance_code="NO2",
        wind_speed=2.0,
        wind_from_degrees=270.0,
        weather_anchors=anchors,
    )
    predictions = anchors.copy()
    predictions["value"] = [0.06, 0.10]

    forecast = build_semiphysical_forecast_grid(
        nowcast,
        predictions,
        substance_code="NO2",
        wind_speed=2.0,
        wind_from_degrees=270.0,
    )

    assert not forecast.empty
    assert set(forecast["source_station_count"]) == {2}
    assert forecast["correction_value"].abs().max() > 0.0


def test_semiphysical_forecast_stores_physical_and_statistical_components():
    anchors = _anchors().iloc[:3].copy()
    analysis = assimilate_observations(
        anchors,
        None,
        wind_speed=2.0,
        wind_from_degrees=270.0,
    )
    forecast_anchors = anchors.copy()
    forecast_anchors["value"] = [0.05, 0.09, 0.13]
    forecast = build_semiphysical_forecast_grid(
        analysis,
        forecast_anchors,
        substance_code="NO2",
        wind_speed=2.0,
        wind_from_degrees=270.0,
        boundary_layer_height=600.0,
        precipitation=0.2,
    )

    assert not forecast.empty
    assert forecast["physical_forecast"].notna().all()
    assert forecast["correction_value"].notna().all()
    assert forecast["analysis_value"].isna().all()
    assert (forecast["diffusion_coefficient"] > 0).all()
    assert (forecast["decay_coefficient"] > 0).all()
    assert (forecast["value"] >= 0).all()


def test_analysis_spreads_forecast_error_upwind():
    anchor = _anchors().iloc[:1].copy()
    anchor["weather_wind_speed"] = 4.0
    anchor["weather_wind_direction"] = 270.0
    initial = anchor.copy()
    initial["value"] = 0.02
    prior = build_semiphysical_forecast_grid(
        assimilate_observations(initial, None),
        initial,
        substance_code="NO2",
        wind_speed=0.0,
        wind_from_degrees=270.0,
    )
    anchor["value"] = 0.10

    analysis = assimilate_observations(anchor, prior)
    latitude = float(anchor.iloc[0].latitude)
    longitude = float(anchor.iloc[0].longitude)
    west_latitude, west_longitude = _offset(latitude, longitude, -8_000.0, 0.0)
    east_latitude, east_longitude = _offset(latitude, longitude, 8_000.0, 0.0)
    west = _nearest_row(analysis, west_latitude, west_longitude)
    east = _nearest_row(analysis, east_latitude, east_longitude)

    assert west["correction_value"] > east["correction_value"]


def test_analysis_preserves_previous_forecast_footprint():
    anchors = _anchors().iloc[:1].copy()
    initial_analysis = assimilate_observations(anchors, None)
    prior_forecast = build_semiphysical_forecast_grid(
        initial_analysis,
        anchors,
        substance_code="NO2",
        wind_speed=4.0,
        wind_from_degrees=270.0,
    )

    next_analysis = assimilate_observations(anchors, prior_forecast)

    assert next_analysis["east"].max() >= prior_forecast["east"].max() - 0.0001
