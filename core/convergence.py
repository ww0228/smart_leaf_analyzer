"""Logistic Regression solver-convergence auditing utilities.

These helpers wrap existing, unmodified `.fit()` calls to record whether the underlying
`LogisticRegression` solver converged -- `n_iter_`, whether it reached `max_iter`, and whether
sklearn raised `ConvergenceWarning` -- without changing what is fit, how it is fit, or any
prediction/metric downstream of it. Nothing in this module fits, refits, or clones an estimator;
it only observes an estimator that the caller already fit and reads its post-fit attributes.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CONVERGENCE_AUDIT_COLUMNS = [
    "Stage", "Seed", "Outer_Fold", "Scenario", "Best_C", "Solver", "Tol",
    "Max_Iter", "N_Iter", "Hit_Max_Iter", "Convergence_Warning",
]


def extract_logistic_regression(estimator: Any) -> LogisticRegression:
    """Return the fitted LogisticRegression step from a bare estimator or a Pipeline.

    Pipelines built by this project name the classifier step "clf" (see
    core.model.build_locked_final_pipeline and the notebook's get_models_and_grids); falls back
    to the last pipeline step if that name is absent.
    """
    candidate = estimator
    if isinstance(estimator, Pipeline):
        candidate = estimator.named_steps.get("clf", estimator[-1])
    if not isinstance(candidate, LogisticRegression):
        raise TypeError(
            "Expected a fitted LogisticRegression (bare, or as a Pipeline step named 'clf'), "
            f"got {type(candidate)!r}."
        )
    if not hasattr(candidate, "n_iter_"):
        raise AssertionError("LogisticRegression estimator has not been fit yet (missing n_iter_).")
    return candidate


@contextmanager
def catch_convergence_warning() -> Iterator[dict[str, Any]]:
    """Observe whether a sklearn ConvergenceWarning fires inside this block.

    Wrap an existing, unmodified `.fit(...)` call in this context manager to capture its
    warnings without changing what is fit or how:

        with catch_convergence_warning() as state:
            estimator.fit(X, y)
        # state["triggered"] is True if a ConvergenceWarning fired during that fit.

    Yields a dict with "triggered" (bool) and "messages" (list[str]), populated when the
    block exits. Uses warnings.catch_warnings(), which saves and restores the global warning
    filter state, so it has no effect on warnings raised outside this block.
    """
    state: dict[str, Any] = {"triggered": False, "messages": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        yield state
    for caught_warning in caught:
        if issubclass(caught_warning.category, ConvergenceWarning):
            state["triggered"] = True
            state["messages"].append(str(caught_warning.message))


def build_convergence_record(*,
                             stage: str,
                             seed: Any,
                             outer_fold: Any,
                             scenario: Any,
                             estimator: Any,
                             convergence_triggered: bool,
                             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one convergence-audit row from an already-fitted estimator.

    Reads attributes off ``estimator`` only -- never fits, refits, or clones it. ``outer_fold``
    should be None for single fixed-seed fits that are not part of an outer CV loop (e.g. the
    final deployed fit, or a Final-holdout scenario fit).
    """
    clf = extract_logistic_regression(estimator)
    n_iter_value = int(np.max(np.asarray(clf.n_iter_)))
    max_iter_value = int(clf.max_iter)
    record: dict[str, Any] = {
        "Stage": stage,
        "Seed": seed,
        "Outer_Fold": outer_fold,
        "Scenario": scenario,
        "Best_C": float(clf.C),
        "Solver": clf.solver,
        "Tol": float(clf.tol),
        "Max_Iter": max_iter_value,
        "N_Iter": n_iter_value,
        "Hit_Max_Iter": bool(n_iter_value >= max_iter_value),
        "Convergence_Warning": bool(convergence_triggered),
    }
    if extra:
        record.update(extra)
    return record


def summarize_convergence_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    """One-row summary: total fits, min/median/max N_Iter, max_iter-hit and warning counts."""
    if audit_df.empty:
        raise AssertionError("Convergence audit table is empty; nothing to summarize.")
    n_iter = audit_df["N_Iter"].astype(int)
    return pd.DataFrame([{
        "Total_Fits": int(len(audit_df)),
        "Min_N_Iter": int(n_iter.min()),
        "Median_N_Iter": float(n_iter.median()),
        "Max_N_Iter": int(n_iter.max()),
        "Fits_Hitting_Max_Iter": int(audit_df["Hit_Max_Iter"].sum()),
        "Fits_With_Convergence_Warning": int(audit_df["Convergence_Warning"].sum()),
    }])


def check_convergence_audit(audit_df: pd.DataFrame) -> bool:
    """Warn (non-fatal) if any audited fit reached max_iter or raised ConvergenceWarning.

    Returns True if every audited fit converged cleanly, False otherwise. This function never
    raises on a genuine non-convergence finding -- surfacing that finding is the point of the
    audit, not a bug in the audit -- and it never fits, refits, or otherwise changes any model.
    """
    hit_max_iter = audit_df.loc[audit_df["Hit_Max_Iter"]]
    warned = audit_df.loc[audit_df["Convergence_Warning"]]
    ok = hit_max_iter.empty and warned.empty
    if not ok:
        message = (
            f"Logistic Regression convergence audit found {len(hit_max_iter)} fit(s) reaching "
            f"max_iter and {len(warned)} fit(s) raising ConvergenceWarning. "
            "See the convergence audit table for Stage/Seed/Outer_Fold/Scenario detail."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {message}")
    return ok
