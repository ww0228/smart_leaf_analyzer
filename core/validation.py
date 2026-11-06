"""Quality and column checks shared by the notebook and dashboard.

These functions perform wavelength-column detection and preprocessing quality checks. 
They raise AssertionError or ValueError when invalid data or unexpected processing results are detected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.preprocessing import EDA_FEATURE_COLS, REIP_MIN, REIP_MAX, WAVELENGTH_MAX, WAVELENGTH_MIN


def is_wavelength_column(column_name: Any) -> bool:
    try:
        float(str(column_name).strip())
    except (TypeError, ValueError):
        return False
    return True


def detect_wavelength_columns(df: pd.DataFrame,
                              wl_min: int = WAVELENGTH_MIN,
                              wl_max: int = WAVELENGTH_MAX) -> list[str]:
    pairs = []
    for col in df.columns:
        if not is_wavelength_column(col):
            continue
        wavelength = float(str(col).strip())
        if wl_min <= wavelength <= wl_max:
            pairs.append((wavelength, col))
    pairs = sorted(pairs, key=lambda item: item[0])
    return [col for _, col in pairs]


def assert_quality_checks(features: pd.DataFrame,
                          raw: np.ndarray,
                          normalized: np.ndarray,
                          wavelengths: np.ndarray) -> None:
    assert len(wavelengths) == 71, f"Expected 71 bands from 500-850 nm, got {len(wavelengths)}."
    assert int(wavelengths.min()) == 500 and int(wavelengths.max()) == 850
    assert not np.isnan(raw).any(), "Missing spectral values found after loading."
    assert np.allclose(np.nanmin(normalized, axis=1), 0.0), "Some normalized rows do not start at 0."
    assert np.allclose(np.nanmax(normalized, axis=1), 1.0), "Some normalized rows do not end at 1."
    assert features["REIP_nm"].between(REIP_MIN, REIP_MAX).all(), "Some REIP values are outside 680-750 nm."
    assert np.isfinite(features[EDA_FEATURE_COLS].to_numpy()).all(), "Non-finite feature values found."
    assert features.groupby("Sample_Group_ID")["Dataset_ID"].nunique().max() == 1
