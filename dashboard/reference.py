"""The artefact is exported alongside the locked model and metadata.
The loader validates its structure and feature order, while deployment
must keep all artefacts from the same export bundle together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_PATH = REPO_ROOT / "outputs" / "development_reference.npz"

_CURVE_KEYS = ("healthy_curve_mean", "healthy_curve_sd", "tipburn_curve_mean", "tipburn_curve_sd")
_FEATURE_KEYS = ("healthy_features", "tipburn_features")
_REQUIRED_KEYS = ("wavelengths", *_CURVE_KEYS, *_FEATURE_KEYS, "feature_order", "n_healthy", "n_tipburn")


@dataclass(frozen=True)
class DevelopmentReference:
    """Frozen development-set reference distributions for the explanation view."""

    wavelengths: np.ndarray
    healthy_curve_mean: np.ndarray
    healthy_curve_sd: np.ndarray
    tipburn_curve_mean: np.ndarray
    tipburn_curve_sd: np.ndarray
    healthy_features: np.ndarray   # (n_healthy, n_features)
    tipburn_features: np.ndarray   # (n_tipburn, n_features)
    feature_order: list[str]
    n_healthy: int
    n_tipburn: int


def load_reference_artifact(
    path: Path | str = DEFAULT_REFERENCE_PATH,
    expected_feature_order: list[str] | None = None,
) -> DevelopmentReference:
    """Load and validate the development reference artefact.

    If expected_feature_order is given (e.g. the locked model's feature order), assert it matches
    the artefact's feature_order exactly -- this is the guard against the reference drifting from
    the model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Development reference artefact not found: {path}. "
            "Re-run the notebook's model-export cell to regenerate it."
        )
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"Development reference is missing key(s): {', '.join(missing)}.")
        wavelengths = data["wavelengths"].astype(float)
        curves = {key: data[key].astype(float) for key in _CURVE_KEYS}
        healthy_features = data["healthy_features"].astype(float)
        tipburn_features = data["tipburn_features"].astype(float)
        feature_order = [str(name) for name in data["feature_order"].tolist()]
        n_healthy = int(data["n_healthy"])
        n_tipburn = int(data["n_tipburn"])

    n_bands = wavelengths.shape[0]
    for key, curve in curves.items():
        if curve.shape != (n_bands,):
            raise ValueError(f"Reference curve {key} has shape {curve.shape}, expected ({n_bands},).")
    n_features = len(feature_order)
    if healthy_features.shape[1] != n_features or tipburn_features.shape[1] != n_features:
        raise ValueError("Reference feature matrices do not match feature_order width.")
    if healthy_features.shape[0] != n_healthy or tipburn_features.shape[0] != n_tipburn:
        raise ValueError("Reference feature-matrix row counts do not match n_healthy / n_tipburn.")
    if expected_feature_order is not None and feature_order != list(expected_feature_order):
        raise AssertionError(
            f"Reference feature_order {feature_order} does not match the locked model "
            f"feature order {list(expected_feature_order)}."
        )

    return DevelopmentReference(
        wavelengths=wavelengths,
        healthy_curve_mean=curves["healthy_curve_mean"],
        healthy_curve_sd=curves["healthy_curve_sd"],
        tipburn_curve_mean=curves["tipburn_curve_mean"],
        tipburn_curve_sd=curves["tipburn_curve_sd"],
        healthy_features=healthy_features,
        tipburn_features=tipburn_features,
        feature_order=feature_order,
        n_healthy=n_healthy,
        n_tipburn=n_tipburn,
    )


__all__ = ["DevelopmentReference", "DEFAULT_REFERENCE_PATH", "load_reference_artifact"]
