"""Operational leaf-level spectral preprocessing.

This module is the implementation authority shared by model development
and dashboard inference. Changes to the wavelength range, smoothing
parameters, normalisation, feature formulas, or feature order require
revalidation and regeneration of the locked model artefacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

EPS = 1e-12

WAVELENGTH_MIN = 500
WAVELENGTH_MAX = 850
REIP_MIN = 680
REIP_MAX = 750
NIR_MIN = 750
NIR_MAX = 850

DEFAULT_WINDOW = 11
POLYORDER = 2

CONTEXT_COLS = ["Dataset_ID", "Label", "Week", "DAT"]
ID_COLS = ["Lettuce_Sample_ID", "Leaf_ID", "Area_ID"]
LEAF_ID_COLS = ["Sample_Group_ID", "Lettuce_Sample_ID", "Leaf_ID"]

EDA_FEATURE_COLS = [
    "REIP_nm",
    "NDRE",
    "Red_Edge_Slope",
    "NDVI",
    "NIR_Mean_Norm",
    "NIR_Mean_SmoothedRaw",
]

FINAL_FEATURE_COLS = [
    "REIP_nm",
    "NDRE",
    "Red_Edge_Slope",
    "NDVI",
    "NIR_Mean_SmoothedRaw",
]

LABEL_TO_BINARY = {
    "Healthy": 0, "healthy": 0, "Control": 0, "control": 0,
    "Tipburn": 1, "tipburn": 1, 0: 0, 1: 1,
}


def group_to_leaf_level(raw_df: pd.DataFrame,
                        wavelength_cols: list[str]) -> pd.DataFrame:
    """Average scans to area means, then area means to one row per leaf."""
    area_keys = [*CONTEXT_COLS, "Sample_Group_ID", *ID_COLS]
    raw_spectral = raw_df[area_keys + wavelength_cols].copy()
    raw_spectral = raw_spectral.dropna(subset=wavelength_cols)

    area_means = (
        raw_spectral
        .groupby(area_keys, dropna=False)[wavelength_cols]
        .mean()
        .reset_index()
    )
    scan_counts = (
        raw_spectral
        .groupby(area_keys, dropna=False)
        .size()
        .rename("Scan_Count")
        .reset_index()
    )
    area_df = area_means.merge(scan_counts, on=area_keys, how="left")

    leaf_keys = [*CONTEXT_COLS, *LEAF_ID_COLS]
    leaf_means = (
        area_df
        .groupby(leaf_keys, dropna=False)[wavelength_cols]
        .mean()
        .reset_index()
    )
    leaf_audit = (
        area_df
        .groupby(leaf_keys, dropna=False)
        .agg(
            Area_Count=("Area_ID", "nunique"),
            Scan_Count_Total=("Scan_Count", "sum"),
            Scan_Count_Min=("Scan_Count", "min"),
            Scan_Count_Median=("Scan_Count", "median"),
            Scan_Count_Max=("Scan_Count", "max"),
        )
        .reset_index()
    )
    leaf_df = leaf_means.merge(leaf_audit, on=leaf_keys, how="left")
    return leaf_df


def rowwise_minmax(values: np.ndarray, epsilon: float = EPS) -> np.ndarray:
    row_min = np.nanmin(values, axis=1, keepdims=True)
    row_max = np.nanmax(values, axis=1, keepdims=True)
    span = row_max - row_min
    zero_span = span < epsilon
    span = np.where(zero_span, 1.0, span)
    normalized = (values - row_min) / span
    normalized = np.where(zero_span, 0.0, normalized)
    return normalized


def interpolate_each_row(values: np.ndarray,
                         wavelengths: np.ndarray,
                         target_nm: float) -> np.ndarray:
    return np.array([np.interp(target_nm, wavelengths, row) for row in values])


def preprocess_leaf_spectra(leaf_df: pd.DataFrame,
                            wavelength_cols: list[str],
                            wavelengths: np.ndarray) -> dict[str, Any]:
    raw = leaf_df[wavelength_cols].to_numpy(dtype=float)
    smoothed = savgol_filter(
        raw,
        window_length=DEFAULT_WINDOW,
        polyorder=POLYORDER,
        axis=1,
        mode="interp",
    )
    normalized = rowwise_minmax(smoothed)
    derivative = np.gradient(normalized, wavelengths, axis=1)

    red_edge_mask = (wavelengths >= REIP_MIN) & (wavelengths <= REIP_MAX)
    nir_mask = (wavelengths >= NIR_MIN) & (wavelengths <= NIR_MAX)
    red_edge_wavelengths = wavelengths[red_edge_mask]
    red_edge_derivative = derivative[:, red_edge_mask]
    reip_idx = np.argmax(red_edge_derivative, axis=1)

    r720 = interpolate_each_row(smoothed, wavelengths, 720)
    r750 = interpolate_each_row(smoothed, wavelengths, 750)
    r670 = interpolate_each_row(smoothed, wavelengths, 670)
    r800 = interpolate_each_row(smoothed, wavelengths, 800)

    features = leaf_df[
        [
            *CONTEXT_COLS,
            *LEAF_ID_COLS,
            "Area_Count",
            "Scan_Count_Total",
            "Scan_Count_Min",
            "Scan_Count_Median",
            "Scan_Count_Max",
        ]
    ].copy()
    features["y"] = features["Label"].map(LABEL_TO_BINARY).astype(int)
    features["REIP_nm"] = red_edge_wavelengths[reip_idx]
    features["Red_Edge_Slope"] = red_edge_derivative[np.arange(len(features)), reip_idx]
    features["NDRE"] = (r750 - r720) / (r750 + r720 + EPS)
    features["NDVI"] = (r800 - r670) / (r800 + r670 + EPS)
    features["NIR_Mean_Norm"] = normalized[:, nir_mask].mean(axis=1)
    features["NIR_Mean_SmoothedRaw"] = smoothed[:, nir_mask].mean(axis=1)

    return {
        "raw": raw,
        "smoothed": smoothed,
        "normalized": normalized,
        "derivative": derivative,
        "features": features,
    }
