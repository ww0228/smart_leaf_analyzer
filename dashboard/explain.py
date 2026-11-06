"""Compute per-leaf contributions for the locked Logistic Regression model."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LeafContributions:
    """Linear decomposition of one leaf's locked-model decision."""

    feature_order: list[str]
    standardized_values: np.ndarray
    coefficients: np.ndarray
    contributions: np.ndarray
    intercept: float
    logit: float
    score: float


def _sigmoid(value: float) -> float:
    """Convert the linear model value into the displayed model score."""

    return float(
        1.0 / (
            1.0 + np.exp(-value)
        )
    )


def feature_contributions(
    pipeline,
    x_row: np.ndarray,
    feature_order: list[str],
) -> LeafContributions:
    """Decompose one leaf's Logistic Regression decision."""
    
    scaler = pipeline.named_steps[
        "scaler"
    ]

    classifier = pipeline.named_steps[
        "clf"
    ]

    x = np.asarray(
        x_row,
        dtype=float,
    ).reshape(-1)

    if x.shape[0] != len(
        feature_order
    ):
        raise ValueError(
            f"x_row has {x.shape[0]} values, "
            f"expected {len(feature_order)}."
        )

    standardized_values = (
        x - scaler.mean_
    ) / scaler.scale_

    coefficients = (
        classifier.coef_[0]
    )

    contributions = (
        standardized_values
        * coefficients
    )

    intercept = float(
        classifier.intercept_[0]
    )

    logit = float(
        contributions.sum()
        + intercept
    )

    return LeafContributions(
        feature_order=list(
            feature_order
        ),
        standardized_values=standardized_values,
        coefficients=coefficients,
        contributions=contributions,
        intercept=intercept,
        logit=logit,
        score=_sigmoid(logit),
    )


__all__ = [
    "LeafContributions",
    "feature_contributions",
]