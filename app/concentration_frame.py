from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import numpy as np
import pandas as pd


FRAME_FORMAT_VERSION = 1
FRAME_COLUMNS = {
    "grid_x": np.int32,
    "grid_y": np.int32,
    "latitude": np.float64,
    "longitude": np.float64,
    "south": np.float64,
    "west": np.float64,
    "north": np.float64,
    "east": np.float64,
    "value": np.float32,
    "analysis_value": np.float32,
    "physical_forecast": np.float32,
    "correction_value": np.float32,
    "lower_bound": np.float32,
    "upper_bound": np.float32,
    "confidence": np.float32,
    "source_station_count": np.int16,
    "wind_speed": np.float32,
    "wind_direction": np.float32,
    "wind_u": np.float32,
    "wind_v": np.float32,
    "boundary_layer_height": np.float32,
    "diffusion_coefficient": np.float32,
    "decay_coefficient": np.float32,
}


def encode_frame(frame: pd.DataFrame) -> tuple[bytes, str]:
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(FRAME_FORMAT_VERSION, dtype=np.int16),
    }
    for column, dtype in FRAME_COLUMNS.items():
        if column in frame:
            values = frame[column].to_numpy()
        elif np.issubdtype(dtype, np.floating):
            values = np.full(len(frame), np.nan)
        else:
            raise ValueError(f"Concentration frame is missing required column: {column}")
        arrays[column] = np.asarray(values, dtype=dtype)

    output = BytesIO()
    np.savez_compressed(output, **arrays)
    content = output.getvalue()
    return content, sha256(content).hexdigest()


def decode_frame(content: bytes, expected_hash: str | None = None) -> pd.DataFrame:
    if expected_hash is not None and sha256(content).hexdigest() != expected_hash:
        raise ValueError("Concentration frame checksum does not match metadata")
    with np.load(BytesIO(content), allow_pickle=False) as archive:
        version = int(archive["format_version"])
        if version != FRAME_FORMAT_VERSION:
            raise ValueError(f"Unsupported concentration frame format: {version}")
        return pd.DataFrame({column: archive[column] for column in FRAME_COLUMNS})
