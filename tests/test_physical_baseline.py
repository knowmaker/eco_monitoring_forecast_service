import pandas as pd

from app.features.physical_baseline import add_physical_baseline


def test_physical_baseline_is_nonnegative_and_uses_upwind_value():
    frame = pd.DataFrame({
        "substance_code": ["NO2"],
        "gas_NO2_lag_0": [1.0],
        "spatial_upwind_value": [3.0],
        "transport_weather_hor_win_spd": [6.0],
        "transport_weather_precipitation": [0.0],
        "source_weather_boundary_layer_height": [500.0],
        "target_weather_boundary_layer_height": [500.0],
    })

    result = add_physical_baseline(frame)

    assert result.loc[0, "physical_baseline"] > 1.0
    assert result.loc[0, "physical_baseline"] < 3.0
