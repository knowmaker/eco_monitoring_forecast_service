from app.config import SERVICE_ROOT
from app.training.model_store import resolve_artifact_path


def test_resolve_artifact_path_accepts_windows_separator_from_database():
    resolved = resolve_artifact_path(r"artifacts\current\gas_forecast.joblib")

    assert resolved == SERVICE_ROOT / "artifacts" / "current" / "gas_forecast.joblib"
