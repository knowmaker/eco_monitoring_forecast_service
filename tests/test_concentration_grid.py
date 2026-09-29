import numpy as np
import pandas as pd

from app.features.concentration_grid import (
    assimilate_observations,
    build_concentration_grid,
    build_semiphysical_forecast_grid,
)


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


def test_grid_builds_separate_area_for_isolated_station():
    grid = build_concentration_grid(
        _anchors(),
        data_kind="observed",
        wind_speed=2.0,
        wind_from_degrees=270.0,
    )

    assert not grid.empty
    assert set(grid["cluster_id"]) == {1, 4}
    assert set(grid["source_station_count"]) == {1, 3}
    assert grid["confidence"].between(0.0, 1.0).all()
    assert (grid["south"] < grid["north"]).all()
    assert (grid["west"] < grid["east"]).all()


def test_forecast_grid_preserves_nonnegative_values_and_intervals():
    grid = build_concentration_grid(
        _anchors().iloc[:3],
        data_kind="forecast",
        wind_speed=3.0,
        wind_from_degrees=180.0,
    )

    assert (grid["value"] >= 0.0).all()
    assert (grid["lower_bound"] <= grid["value"]).all()
    assert (grid["upper_bound"] >= grid["value"]).all()


def test_forecast_grid_extends_in_transport_direction():
    anchors = _anchors().iloc[:3].copy()
    source = anchors.copy()
    source["value"] = [0.12, 0.02, 0.02]

    calm_grid = build_concentration_grid(
        anchors,
        data_kind="forecast",
        source_anchors=source,
    )
    wind_grid = build_concentration_grid(
        anchors,
        data_kind="forecast",
        wind_speed=4.0,
        wind_from_degrees=270.0,
        source_anchors=source,
    )

    assert wind_grid["east"].max() > calm_grid["east"].max()
    assert wind_grid.loc[wind_grid["longitude"] > calm_grid["east"].max(), "value"].notna().any()


def test_forecast_grid_uses_predictions_when_recent_observations_are_missing():
    anchors = _anchors().iloc[:3].copy()

    calm_grid = build_concentration_grid(anchors, data_kind="forecast")
    wind_grid = build_concentration_grid(
        anchors,
        data_kind="forecast",
        wind_speed=4.0,
        wind_from_degrees=270.0,
    )

    assert wind_grid["east"].max() > calm_grid["east"].max()


def test_grid_clips_negative_concentrations_instead_of_creating_mass():
    anchors = _anchors().iloc[:1].copy()
    anchors["value"] = -0.12

    grid = build_concentration_grid(anchors, data_kind="observed")

    assert (grid["value"] >= 0.0).all()
    assert grid["value"].max() == 0.0
    assert grid["value"].nunique() == 1
    assert (grid["north"].max() - anchors.iloc[0].latitude) * 111_320 > 900


def test_single_station_grid_has_rounded_coverage():
    anchors = _anchors().iloc[:1].copy()

    grid = build_concentration_grid(anchors, data_kind="observed")

    latitude = float(anchors.iloc[0].latitude)
    longitude = float(anchors.iloc[0].longitude)
    northing = (grid["latitude"] - latitude) * 111_320.0
    easting = (
        (grid["longitude"] - longitude)
        * 111_320.0
        * np.cos(np.deg2rad(latitude))
    )
    distances = np.hypot(easting, northing)
    assert distances.max() <= 1500.01
    assert len(grid) < 31 * 31


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
