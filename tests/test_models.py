import pandas as pd

from app.features.spatial_features import add_spatial_features
from app.models.base import ModelArtifact, prediction_results


class ConstantEstimator:
    def predict(self, frame):
        return [-0.5, 1.5][: len(frame)]


def test_prediction_results_do_not_return_negative_concentrations():
    artifact = ModelArtifact(ConstantEstimator(), ["feature"], [], -0.2, 0.3, {})
    results = prediction_results(artifact, pd.DataFrame({"feature": [1, 2]}))
    assert results[0].value == 0.0
    assert results[0].lower_bound == 0.0
    assert results[1].value == 1.5
    assert results[1].upper_bound == 1.8


def test_spatial_feature_uses_upwind_neighbor():
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "monitoring_post_id": [1, 2],
            "source_bucket": [timestamp, timestamp],
            "latitude": [55.0, 55.01],
            "longitude": [37.0, 37.0],
            "source_weather_hor_win_dir": [0.0, 180.0],
            "source_weather_hor_win_spd": [5.0, 5.0],
            "gas_NO2_lag_0": [1.0, 3.0],
        }
    )
    spatial = add_spatial_features(frame, "NO2")
    assert spatial.loc[0, "spatial_neighbor_count"] == 1
    assert spatial.loc[0, "spatial_upwind_value"] == 3.0
