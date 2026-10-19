from app.models.base import ModelArtifact, deserialize_artifact, serialize_artifact


class ConstantEstimator:
    def predict(self, frame):
        return [1.5] * len(frame)


def test_model_artifact_round_trip_through_bytes():
    artifact = ModelArtifact(
        estimator=ConstantEstimator(),
        feature_columns=["temperature"],
        categorical_features=[],
        residual_lower=-0.1,
        residual_upper=0.2,
        metadata={"active_gases": ["NO2"]},
    )

    restored = deserialize_artifact(serialize_artifact(artifact))

    assert restored.feature_columns == ["temperature"]
    assert restored.metadata == {"active_gases": ["NO2"]}
