import numpy as np
import pandas as pd
import pytest

from app.concentration_frame import FRAME_COLUMNS, decode_frame, encode_frame


def test_concentration_frame_round_trip():
    source = pd.DataFrame(
        {
            column: (
                np.asarray([1, 2], dtype=dtype)
                if np.issubdtype(dtype, np.integer)
                else np.asarray([0.25, np.nan], dtype=dtype)
            )
            for column, dtype in FRAME_COLUMNS.items()
        }
    )

    content, content_hash = encode_frame(source)
    restored = decode_frame(content, content_hash)

    assert len(content_hash) == 64
    assert list(restored.columns) == list(FRAME_COLUMNS)
    for column in FRAME_COLUMNS:
        np.testing.assert_allclose(restored[column], source[column], equal_nan=True)

    with pytest.raises(ValueError, match="checksum"):
        decode_frame(content, "0" * 64)
