"""Adapter between dashboard scan-level data and the locked core pipeline.

The raw-data workflow assigns Lettuce_Sample_ID, Leaf_ID, and Area_ID
before calling this module. Internal context fields are added only to
satisfy the labelled research preprocessing interface; they are not
user-supplied ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.preprocessing import (  # noqa: E402
    group_to_leaf_level,
    preprocess_leaf_spectra,
)
from core.validation import assert_quality_checks, detect_wavelength_columns  # noqa: E402

REQUIRED_ID_COLS = ["Lettuce_Sample_ID", "Leaf_ID", "Area_ID"]
EXPECTED_BAND_COUNT = 71
PLACEHOLDER_LABEL = "Healthy"
OPTIONAL_CONTEXT_DEFAULTS = {"Dataset_ID": "Uploaded", "Week": pd.NA, "DAT": pd.NA}

# Per-leaf data-quality thresholds (display warnings only -- these never block or drop rows).
QUALITY_ID_COLS = ["Sample_Group_ID", "Lettuce_Sample_ID", "Leaf_ID"]
QUALITY_MIN_AREAS = 3            # a complete leaf is scanned over 3 areas
QUALITY_MIN_SCAN_TOTAL = 15     # development leaves average 30 scans (3 areas x 10); < 15 is thin
QUALITY_REFL_MIN = 0.0          # plausible reflectance-fraction lower bound
QUALITY_REFL_MAX = 1.2          # plausible reflectance-fraction upper bound (small margin over 1.0)
QUALITY_FLAT_STD = 0.01         # per-leaf spectral std below this is implausibly flat


class ValidationError(Exception):
    """Raised when the uploaded CSV cannot be prepared for classification."""


def validate_raw_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems; empty list means the file is usable."""
    problems = []
    missing_ids = [col for col in REQUIRED_ID_COLS if col not in df.columns]
    if missing_ids:
        problems.append(
            f"Missing required column(s): {', '.join(missing_ids)}. "
            "Expected one row per scan with Lettuce_Sample_ID, Leaf_ID, and Area_ID."
        )

    wavelength_cols = detect_wavelength_columns(df)
    if len(wavelength_cols) < EXPECTED_BAND_COUNT:
        problems.append(
            f"Found only {len(wavelength_cols)} wavelength columns in the 500-850 nm range "
            f"(need {EXPECTED_BAND_COUNT}, i.e. 500-850 nm in 5 nm steps). "
            "Check the file has one column per wavelength with numeric headers."
        )
    return problems


def count_missing_spectral_rows(df: pd.DataFrame,) -> int:
    """Count scan rows that preprocessing will remove for missing spectra."""
    wavelength_cols = detect_wavelength_columns(df)

    if not wavelength_cols:
        return 0

    missing_row_mask = df[
        wavelength_cols
    ].isna().any(
        axis=1
    )

    return int(
        missing_row_mask.sum()
    )


def prepare_leaf_level(df: pd.DataFrame,) -> tuple[pd.DataFrame, list[str]]:
    """Add internal context fields and aggregate scans to one row per leaf."""

    problems = validate_raw_columns(df)

    if problems:
        raise ValidationError(
            " ".join(problems)
        )

    working = df.copy()

    # Required internally by the shared labelled preprocessing interface.
    working["Label"] = PLACEHOLDER_LABEL

    if "Sample_Group_ID" not in working.columns:
        working["Sample_Group_ID"] = (
            working["Lettuce_Sample_ID"]
        )

    for col, default in (
        OPTIONAL_CONTEXT_DEFAULTS.items()
    ):
        if col not in working.columns:
            working[col] = default

    wavelength_cols = detect_wavelength_columns(
        working
    )

    leaf_df = group_to_leaf_level(
        working,
        wavelength_cols,
    )

    return leaf_df, wavelength_cols


def compute_features_with_spectra(
    leaf_df: pd.DataFrame, wavelength_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute leaf-level features and return the normalised spectra and wavelength grid.

    Uses the locked preprocess_leaf_spectra implementation. The normalised
    spectra are returned for the per-leaf explanation view and remain aligned
    row-for-row with the feature table.
    """
    wavelengths = np.array([float(col) for col in wavelength_cols])
    result = preprocess_leaf_spectra(leaf_df, wavelength_cols, wavelengths)
    features = result["features"]
    normalized = result["normalized"]
    assert_quality_checks(features, result["raw"], normalized, wavelengths)
    features = features.drop(columns=["y"], errors="ignore")
    if normalized.shape[0] != len(features):
        raise AssertionError("Normalized spectra are not aligned with the features table.")
    return features, normalized, wavelengths


def compute_quality_flags(
    leaf_df: pd.DataFrame,
    wavelength_cols: list[str],
    *,
    min_areas: int = QUALITY_MIN_AREAS,
    min_scan_total: int = QUALITY_MIN_SCAN_TOTAL,
    refl_min: float = QUALITY_REFL_MIN,
    refl_max: float = QUALITY_REFL_MAX,
    flat_std: float = QUALITY_FLAT_STD,
) -> pd.DataFrame:
    """Return warning-only leaf-level quality diagnostics.

    The function does not remove or reorder rows. An empty Quality_Flag
    indicates that no warning condition was detected.
    """
    spectra = leaf_df[wavelength_cols].to_numpy(dtype=float)
    refl_min_per_leaf = np.nanmin(spectra, axis=1)
    refl_max_per_leaf = np.nanmax(spectra, axis=1)
    refl_std_per_leaf = np.nanstd(spectra, axis=1)

    records = []
    for position, (_, leaf) in enumerate(leaf_df.reset_index(drop=True).iterrows()):
        reasons: list[str] = []

        area_count = leaf.get("Area_Count")
        if pd.notna(area_count) and float(area_count) < min_areas:
            reasons.append(f"incomplete: fewer than {min_areas} areas")

        scan_total = leaf.get("Scan_Count_Total")
        if pd.notna(scan_total) and float(scan_total) < min_scan_total:
            reasons.append("thin averaging")

        low = refl_min_per_leaf[position]
        high = refl_max_per_leaf[position]
        std = refl_std_per_leaf[position]
        capture_issues: list[str] = []
        if low < refl_min or high > refl_max:
            capture_issues.append("reflectance out of plausible range")
        if std < flat_std:
            capture_issues.append("near-flat spectrum")
        if capture_issues:
            reasons.append("possible bad capture (" + ", ".join(capture_issues) + ")")

        record = {col: leaf.get(col) for col in QUALITY_ID_COLS if col in leaf_df.columns}
        record["Area_Count"] = area_count
        record["Scan_Count_Total"] = scan_total
        record["Reflectance_Min"] = float(low)
        record["Reflectance_Max"] = float(high)
        record["Reflectance_Std"] = float(std)
        record["Quality_Flag"] = "; ".join(reasons)
        records.append(record)

    return pd.DataFrame.from_records(records)


__all__ = [
    "ValidationError",
    "REQUIRED_ID_COLS",
    "QUALITY_ID_COLS",
    "QUALITY_MIN_AREAS",
    "QUALITY_MIN_SCAN_TOTAL",
    "QUALITY_REFL_MIN",
    "QUALITY_REFL_MAX",
    "QUALITY_FLAT_STD",
    "validate_raw_columns",
    "prepare_leaf_level",
    "compute_features_with_spectra",
    "compute_quality_flags",
    "count_missing_spectral_rows",
]
