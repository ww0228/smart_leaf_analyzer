"""Build, score, and load the locked Logistic Regression pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FINAL_MODEL_NAME = "LogisticRegression"
FINAL_MODEL_C = 10
FINAL_DECISION_THRESHOLD = 0.5

# Resolve model artefacts relative to the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "outputs" 
DEFAULT_MODEL_FILENAME = "final_tipburn_pipeline.joblib"
DEFAULT_METADATA_FILENAME = "final_tipburn_model_metadata.json"


def build_locked_final_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Construct the locked 5-feature Logistic Regression pipeline (unfitted)."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=50, random_state=random_state)),
    ])
    pipeline.set_params(clf__C=FINAL_MODEL_C)
    return pipeline


def continuous_model_score(model: Any, X: np.ndarray) -> np.ndarray:
    """Return probabilities when available, otherwise decision scores."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            # Compatibility fallback for locked LogisticRegression artefacts saved by a newer
            # sklearn: sigmoid(decision_function) is the same binary-class score as predict_proba.
            if hasattr(model, "decision_function"):
                logits = np.asarray(model.decision_function(X))
                if logits.ndim == 1:
                    return 1.0 / (1.0 + np.exp(-logits))
            raise
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if np.ndim(scores) == 2:
            return scores[:, 1]
        return np.asarray(scores)
    return model.predict(X)


def load_locked_pipeline(artifact_dir: Path | str | None = None,
                         model_filename: str = DEFAULT_MODEL_FILENAME,
                         metadata_filename: str = DEFAULT_METADATA_FILENAME) -> tuple[Any, dict[str, Any]]:
    """Load the fitted locked pipeline and its metadata."""
    resolved_dir = Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACT_DIR
    model_path = resolved_dir / model_filename
    metadata_path = resolved_dir / metadata_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Locked pipeline not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    pipeline = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return pipeline, metadata
