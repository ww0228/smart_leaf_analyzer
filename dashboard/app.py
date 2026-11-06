"""Streamlit dashboard for the locked Tipburn classifier.

The dashboard accepts a raw ColorCompass scan-level CSV, assigns each
scan range to its lettuce, leaf, and area, extracts the structured data,
and applies the frozen classification pipeline.

The dashboard never retrains, retunes, or modifies the locked model.
Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.model import (  # noqa: E402
    continuous_model_score,
    load_locked_pipeline,
)
from core.preprocessing import FINAL_FEATURE_COLS
from dashboard.data_prep import (  # noqa: E402
    QUALITY_FLAT_STD,
    QUALITY_ID_COLS,
    QUALITY_MIN_AREAS,
    QUALITY_MIN_SCAN_TOTAL,
    QUALITY_REFL_MAX,
    QUALITY_REFL_MIN,
    ValidationError,
    compute_features_with_spectra,
    compute_quality_flags,
    prepare_leaf_level,
    validate_raw_columns,
    count_missing_spectral_rows,
)
from dashboard.explain import feature_contributions  # noqa: E402
from dashboard.reference import load_reference_artifact  # noqa: E402
from dashboard.raw_data_prep import (  # noqa: E402
    DEFAULT_GAP,
    EDIT_MODE_ORIGINAL,
    EDIT_MODE_SHIFT_FOLLOWING,
    RawDataPrepError,
    apply_range_edit,
    extract_prepared_dataframe,
    generate_manual_ranges,
    validate_ranges,
    validate_raw_upload_columns,
    find_uncovered_row_intervals,
)

CLASS_NAMES = {0: "Healthy", 1: "Tipburn"}


def validate_locked_feature_order(metadata: dict, features: pd.DataFrame) -> list[str]:
    """Return the locked feature order, or raise before any prediction can run."""
    metadata_feature_order = list(metadata.get("feature_order", []))
    core_feature_order = list(FINAL_FEATURE_COLS)
    if metadata_feature_order != core_feature_order:
        raise AssertionError(
            "Model metadata feature_order does not match the dashboard feature order. "
            f"metadata={metadata_feature_order}, dashboard={core_feature_order}"
        )
    missing = [feature for feature in metadata_feature_order if feature not in features.columns]
    if missing:
        raise AssertionError(f"Computed features are missing locked model feature(s): {missing}")
    return metadata_feature_order

st.set_page_config(
    page_title="Smart Leaf Analyzer: Tipburn Classifier",
    layout="wide",
)

def inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }

        .block-container {
            padding-top: 0.6rem;
            padding-bottom: 2.5rem;
        }

        .dashboard-banner {
            padding: 1.1rem 1.6rem;
            margin-top: 5rem;
            margin-bottom: 1.2rem;
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                #1b5e20 0%,
                #388e3c 100%
            );
            color: white;
            box-shadow: 0 8px 24px rgba(27, 94, 32, 0.18);
        }

        .dashboard-banner h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }

        .dashboard-banner p {
            margin: 0.35rem 0 0;
            opacity: 0.92;
            font-size: 1rem;
        }

        .summary-card {
            padding: 1.05rem 1.2rem;
            border-radius: 14px;
            background: rgba(127, 127, 127, 0.08);
            border: 1px solid rgba(127, 127, 127, 0.25);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        }

        .summary-card.healthy {
            border-left: 6px solid #43a047;
        }

        .summary-card.tipburn {
            border-left: 6px solid #c85a55;
        }

        .summary-label {
            margin: 0;
            font-size: 1rem;
            font-weight: 700;
        }

        .summary-value {
            margin: 0.25rem 0 0;
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.05;
            color: inherit;
        }

        .summary-percent {
            margin: 0.3rem 0 0;
            opacity: 0.75;
            font-size: 0.9rem;
        }

        .calculation-box {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background: rgba(127, 127, 127, 0.10);
            border: 1px solid rgba(127, 127, 127, 0.28);
            font-family: ui-monospace, SFMono-Regular, Menlo,
                Monaco, Consolas, monospace;
            font-size: 0.82rem;
            line-height: 1.55;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_title_banner() -> None:
    st.markdown(
        """
        <div class="dashboard-banner">
            <h1>🌿 Smart Leaf Analyzer</h1>
            <p>Spectral classification of Butterhead lettuce leaves</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource(show_spinner=False)
def get_pipeline():
    return load_locked_pipeline()

@st.cache_resource(show_spinner=False)
def get_reference(expected_feature_order: tuple[str, ...]):
    """Load the frozen development reference artefact (cached at the dashboard level)."""
    return load_reference_artifact(expected_feature_order=list(expected_feature_order))


def render_info_panel(metadata: dict) -> None:
    st.sidebar.header("🌿 System Overview")

    st.sidebar.markdown(
        (
            "**Purpose**\n\n"
            "Classifies Butterhead lettuce leaves as Healthy or Tipburn "
            "using spectral measurements.\n\n"
            "**Model**\n\n"
            "- Locked Logistic Regression\n"
            f"- C = {metadata['model_parameters']['clf__C']}\n"
            f"- Decision threshold: {metadata['decision_threshold']}\n"
            "- Five spectral features\n\n"
            "**Analysis Unit**\n\n"
            f"- Prediction unit: one {metadata['prediction_unit']}\n"
            f"- Grouping unit: one {metadata['grouping_unit']}\n\n"
            "**Limitations**\n\n"
            "The dashboard has not been validated for real-time, "
            "pre-visual, or early Tipburn detection."
        )
    )

def render_classification_workflow(
        pipeline,
        metadata,
        raw_df: pd.DataFrame,
    ) -> None:
    """Aggregate, featurise, classify, and display the extracted raw data."""
    st.subheader("Dataset overview")
    sample_count = raw_df["Lettuce_Sample_ID"].nunique() if "Lettuce_Sample_ID" in raw_df.columns else "?"
    st.write(f"{len(raw_df)} scan rows, {sample_count} unique samples.")
    st.dataframe(raw_df.head(20))

    problems = validate_raw_columns(raw_df)
    if problems:
        for problem in problems:
            st.error(problem)
        return

    missing_spectral_row_count = count_missing_spectral_rows(raw_df)

    if missing_spectral_row_count > 0:
        st.warning(
            f"{missing_spectral_row_count} of "
            f"{len(raw_df)} scan rows contain one or more "
            "missing spectral values between 500 and 850 nm. "
            "These rows will be excluded before leaf-level "
            "aggregation."
        )

    try:
        leaf_df, wavelength_cols = prepare_leaf_level(raw_df)
        features, normalized_spectra, feature_wavelengths = compute_features_with_spectra(
            leaf_df, wavelength_cols
        )
    except ValidationError as exc:
        st.error(str(exc))
        return
    except AssertionError as exc:
        st.error(f"Data quality check failed: {exc}")
        return

    st.success(f"Aggregated to {len(features)} leaf-level rows.")

    quality_df = compute_quality_flags(
        leaf_df,
        wavelength_cols,
    )

    flagged = quality_df.loc[
        quality_df["Quality_Flag"] != ""
    ]

    if len(flagged) == 0:
        st.caption(
            "✅ No data-quality warnings detected."
        )
    else:
        st.warning(
            f"{len(flagged)} of {len(quality_df)} "
            "leaves have possible data-quality issues."
        )

        with st.expander(
            "View data-quality warnings"
        ):
            st.caption(
                (
                    "Warnings do not remove leaves or change "
                    "predictions. Checks include fewer than "
                    f"{QUALITY_MIN_AREAS} areas, fewer than "
                    f"{QUALITY_MIN_SCAN_TOTAL} scans, "
                    "reflectance outside "
                    f"[{QUALITY_REFL_MIN}, "
                    f"{QUALITY_REFL_MAX}], or spectral "
                    "standard deviation below "
                    f"{QUALITY_FLAT_STD}."
                )
            )

            st.dataframe(
                flagged[
                    QUALITY_ID_COLS
                    + [
                        "Area_Count",
                        "Scan_Count_Total",
                        "Reflectance_Min",
                        "Reflectance_Max",
                        "Reflectance_Std",
                        "Quality_Flag",
                    ]
                ].round(
                    {
                        "Reflectance_Min": 4,
                        "Reflectance_Max": 4,
                        "Reflectance_Std": 4,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
    results_key = "raw_results"
    if st.button("Run classification", type="primary", key="raw_run_classification"):
        try:
            locked_feature_order = validate_locked_feature_order(metadata, features)
        except AssertionError as exc:
            st.error(f"Locked model feature-order check failed: {exc}")
            return

        X = features[locked_feature_order].to_numpy()
        scores = continuous_model_score(pipeline, X)
        threshold = metadata["decision_threshold"]
        predicted = (scores >= threshold).astype(int)

        results = features[["Sample_Group_ID", "Lettuce_Sample_ID", "Leaf_ID", "Area_Count"]].copy()
        results["Predicted_Class"] = [CLASS_NAMES[p] for p in predicted]
        results["Model_Score_Tipburn"] = scores
        st.session_state[results_key] = results

    if results_key not in st.session_state:
        return

    results = st.session_state[results_key]

    st.subheader("📊 Classification Summary")

    counts = (
        results["Predicted_Class"]
        .value_counts()
        .reindex(["Healthy", "Tipburn"])
        .fillna(0)
        .astype(int)
    )

    total_classified = int(counts.sum())

    if total_classified > 0:
        percentages = (
            counts / total_classified * 100
        ).round(1)
    else:
        percentages = pd.Series(
            {
                "Healthy": 0.0,
                "Tipburn": 0.0,
            }
        )

    healthy_col, tipburn_col = st.columns(
        2,
        gap="medium",
    )

    with healthy_col:
        st.markdown(
            (
                '<div class="summary-card healthy">'
                '<p class="summary-label">🌿 Healthy</p>'
                f'<p class="summary-value">'
                f'{int(counts["Healthy"])}'
                '</p>'
                f'<p class="summary-percent">'
                f'{percentages["Healthy"]:.1f}% '
                'of classified leaves'
                '</p>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    with tipburn_col:
        st.markdown(
            (
                '<div class="summary-card tipburn">'
                '<p class="summary-label">🍂 Tipburn</p>'
                f'<p class="summary-value">'
                f'{int(counts["Tipburn"])}'
                '</p>'
                f'<p class="summary-percent">'
                f'{percentages["Tipburn"]:.1f}% '
                'of classified leaves'
                '</p>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    st.subheader("Results")
    class_filter = st.selectbox(
        "Filter by predicted class", ["All", "Healthy", "Tipburn"], key="raw_filter"
    )
    shown = results if class_filter == "All" else results[results["Predicted_Class"] == class_filter]

    # Build the display-only results table.
    display_df = pd.DataFrame({
        "Lettuce ID": shown["Lettuce_Sample_ID"].to_numpy(),
        "Leaf ID": shown["Leaf_ID"].to_numpy(),
        "Prediction Label": shown["Predicted_Class"].to_numpy(),
        "Model Score": shown["Model_Score_Tipburn"].to_numpy(),
    })

    st.caption(
        "Select a row to view the explanation for that leaf."
    )

    selection_event = st.dataframe(
        display_df.style.format(
            {"Model Score": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="raw_results_select",
    )

    # Map the selected display row back to its leaf record.
    selected_key = None
    try:
        selected_rows = selection_event.selection["rows"] if selection_event is not None else []
    except (AttributeError, KeyError, TypeError):
        selected_rows = []
    if selected_rows:
        _selected_leaf = shown.iloc[int(selected_rows[0])]
        selected_key = (str(_selected_leaf["Sample_Group_ID"]), _selected_leaf["Leaf_ID"])

    # Merge the five extracted model features with the classification results.
    feature_export_cols = (
        QUALITY_ID_COLS + FINAL_FEATURE_COLS
    )

    export_df = results.merge(
        features[feature_export_cols],
        on=QUALITY_ID_COLS,
        how="left",
    )

    st.download_button(
        "Export all results as CSV",
        data=export_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "tipburn_classification_results.csv"
        ),
        mime="text/csv",
        key="raw_export",
    )

    st.caption(
        (
            "Export includes identifiers, the five model "
            "features, prediction label and model score."
        )
    )

    render_leaf_explanation(
        pipeline,
        metadata,
        features,
        normalized_spectra,
        feature_wavelengths,
        selected_key,
    )

RED_EDGE_RANGE = (680.0, 750.0)
NIR_RANGE = (750.0, 850.0)


def render_leaf_explanation(
    pipeline,
    metadata,
    features: pd.DataFrame,
    normalized_spectra,
    feature_wavelengths,
    selected_key,
) -> None:
    """Per-leaf explanation: spectrum, feature positions, contributions, and score.

    Display-only. Loads the frozen development reference (never recomputes it from uploads) and
    reads the fitted scaler/coefficients. Explains the single leaf selected in the Results table
    above (selected_key is (Sample_Group_ID, Leaf_ID), or None when nothing is selected).
    """
    st.subheader("Per-leaf explanation")
    feature_order = list(metadata["feature_order"])
    try:
        reference = get_reference(tuple(feature_order))
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        st.info(f"Per-leaf explanation unavailable: {exc}")
        return

    if selected_key is None:
        st.info("Click a row in the Results table above to explain that leaf.")
        return
    sid, leaf_id = selected_key

    match = np.flatnonzero(
        (features["Sample_Group_ID"].astype(str) == sid) & (features["Leaf_ID"] == leaf_id)
    )
    if len(match) == 0:
        st.info("Could not locate the selected leaf's spectrum.")
        return
    pos = int(match[0])
    leaf_curve = np.asarray(normalized_spectra[pos], dtype=float)
    leaf_feature_values = features.iloc[pos][feature_order].to_numpy(dtype=float)

    # Decompose the selected leaf's Logistic Regression score into
    # standardised feature contributions and the fitted intercept.
    decomposition = feature_contributions(pipeline, leaf_feature_values, feature_order)
    threshold = metadata["decision_threshold"]
    leaf_score = decomposition.score
    predicted_class = CLASS_NAMES[int(leaf_score >= threshold)]
    lettuce_id = features.iloc[pos]["Lettuce_Sample_ID"]
    relation = "below" if leaf_score < threshold else "at or above"

    st.markdown(
        (
            f"**Explaining Lettuce {lettuce_id}, Leaf {leaf_id}**\n\n"
            f"**Classification: {predicted_class}**  \n"
            f"Model score: **{leaf_score:.3f}**  \n"
            f"The score is {relation} the decision threshold of "
            f"**{threshold}**."
        )
    )

    # Spectrum comparison
    st.markdown("**1. Spectrum comparison**")
    fig1, ax1 = plt.subplots(figsize=(7.5, 3.2))
    ax1.axvspan(*RED_EDGE_RANGE, color="#d9a441", alpha=0.10, label="Red edge (680-750 nm)")
    ax1.axvspan(*NIR_RANGE, color="#7fb069", alpha=0.10, label="NIR (750-850 nm)")
    for mean, sd, colour, name in [
        (reference.healthy_curve_mean, reference.healthy_curve_sd, "#2f8f4e", f"Dev Healthy mean (n={reference.n_healthy})"),
        (reference.tipburn_curve_mean, reference.tipburn_curve_sd, "#c0504d", f"Dev Tipburn mean (n={reference.n_tipburn})"),
    ]:
        ax1.plot(reference.wavelengths, mean, color=colour, lw=1.4, alpha=0.7, label=name)
        ax1.fill_between(reference.wavelengths, mean - sd, mean + sd, color=colour, alpha=0.12)
    ax1.plot(feature_wavelengths, leaf_curve, color="#1f1f1f", lw=2.2, label="This leaf")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Normalised reflectance")
    ax1.legend(fontsize=7, loc="upper left")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    st.caption(
        "The black line is the selected leaf. The green and red lines show the "
        "average development spectra for Healthy and Tipburn leaves."
    )

    st.markdown("**2. Extracted feature values**")

    feature_table = pd.DataFrame(
        {
            "Feature": feature_order,
            "Extracted value": leaf_feature_values,
        }
    )

    st.dataframe(
        feature_table.style.format(
            {"Extracted value": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Feature comparison
    st.markdown("**3. Feature comparison**")
    reip_idx = feature_order.index("REIP_nm") if "REIP_nm" in feature_order else None
    width_ratios = [1.35 if col_idx == reip_idx else 1.0 for col_idx in range(len(feature_order))]
    fig2, axes = plt.subplots(
        1,
        len(feature_order),
        figsize=(3.0 * len(feature_order), 3.2),
        gridspec_kw={"width_ratios": width_ratios},
    )
    for col_idx, (ax, feature_name) in enumerate(zip(axes, feature_order)):
        healthy_vals = reference.healthy_features[:, col_idx]
        tipburn_vals = reference.tipburn_features[:, col_idx]
        leaf_value = leaf_feature_values[col_idx]
        if col_idx == reip_idx:
            # REIP is quantised to the 5 nm wavelength grid -> a boxplot over ~3 discrete levels
            # would collapse into near-identical one-step boxes. Instead show the class
            # composition at each grid value as a 100%-stacked bar (Healthy vs Tipburn share),
            # with n printed under each tick so a value backed by few leaves does not look as
            # authoritative as one backed by many. Everything computed from the reference.
            grid = np.unique(np.concatenate([healthy_vals, tipburn_vals]))
            x = np.arange(len(grid))
            h_share = np.zeros(len(grid))
            t_share = np.zeros(len(grid))
            n_per = np.zeros(len(grid), dtype=int)
            for k, g in enumerate(grid):
                n_h = int(np.sum(healthy_vals == g))
                n_t = int(np.sum(tipburn_vals == g))
                n = n_h + n_t
                n_per[k] = n
                if n == 0:
                    continue  # guard against division by zero
                h_share[k] = n_h / n * 100.0
                t_share[k] = n_t / n * 100.0
            # Highlight the column matching this leaf's REIP rather than an axvline, which does
            # not land on a categorical axis.
            sel = int(np.argmin(np.abs(grid - leaf_value)))
            ax.bar(x, h_share, width=0.7, color="#2f8f4e", edgecolor="none", zorder=2)
            ax.bar(x, t_share, width=0.7, bottom=h_share, color="#c0504d", edgecolor="none", zorder=2)
            this_leaf_bar = ax.bar([x[sel]], [100], width=0.7, color="none", zorder=3).patches[0]
            this_leaf_bar.set_edgecolor("#1f1f1f")
            this_leaf_bar.set_linewidth(2.5)
            for k in range(len(grid)):
                if n_per[k] == 0:
                    continue
                if t_share[k] > 0:
                    ax.text(x[k], h_share[k] + t_share[k] / 2, f"{t_share[k]:.0f}% T",
                            ha="center", va="center", fontsize=6, color="white")
                if h_share[k] > 0:
                    ax.text(x[k], h_share[k] / 2, f"{h_share[k]:.0f}% H",
                            ha="center", va="center", fontsize=6, color="white")
            if h_share[sel] >= t_share[sel]:
                label_y = h_share[sel] / 2
                segment_top = h_share[sel]
            else:
                label_y = h_share[sel] + t_share[sel] / 2
                segment_top = 100.0
            ax.scatter(
                [x[sel]],
                [min(label_y + 10.0, segment_top - 4.0)],
                marker="*",
                s=180,
                color="#1f1f1f",
                clip_on=False,
                zorder=5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{g:.0f}\nn = {n_per[k]}" for k, g in enumerate(grid)],
                fontsize=6,
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of dev leaves")
        else:
            bp = ax.boxplot(
                [healthy_vals, tipburn_vals],
                positions=[0, 1],
                widths=0.5,
                showfliers=True,
                flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#999",
                                markeredgecolor="none", alpha=0.6),
                patch_artist=True,
                boxprops=dict(edgecolor="#333", lw=1.0),
                medianprops=dict(color="#1f1f1f", lw=1.6),
                whiskerprops=dict(color="#555", lw=1.0),
                capprops=dict(color="#555", lw=1.0),
                zorder=1,
            )
            for patch, colour in zip(bp["boxes"], ["#2f8f4e", "#c0504d"]):
                patch.set_facecolor(colour)
                patch.set_alpha(0.55)
            # This leaf's value -- the whole point of the panel.
            ax.axhline(leaf_value, color="#1f1f1f", lw=2.0, ls="--")
            ax.scatter([0.5], [leaf_value], marker="*", s=180, color="#1f1f1f", zorder=5)
            # boxplot resets ticks, so set these afterwards.
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Healthy", "Tipburn"], fontsize=7)
        ax.set_title(feature_name, fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption(
        "The star and dashed line mark the selected leaf. Overlap between the "
        "Healthy and Tipburn distributions can lead to difficult or incorrect "
        "classifications."
    )

    # Classification contributions
    st.markdown("**4. Classification decision**")
    order = np.argsort(decomposition.contributions)
    contrib_sorted = decomposition.contributions[order]
    names_sorted = [feature_order[i] for i in order]
    y_features = np.arange(len(names_sorted)) + 1.0
    y_net = -0.4
    fig3, ax3 = plt.subplots(figsize=(7.0, 3.4))
    ax3.barh(y_features, contrib_sorted, color=["#c0504d" if c >= 0 else "#2f8f4e" for c in contrib_sorted])
    # Net bar (sum of contributions + intercept) -- neutral grey so it reads as a total, not a feature.
    ax3.barh([y_net], [decomposition.logit], color="#777")
    ax3.axhline(0.3, color="#999", lw=0.8, ls=":")  # divider between feature bars and the net
    ax3.axvline(0, color="#555", lw=0.8)
    ax3.set_yticks(list(y_features) + [y_net])
    ax3.set_yticklabels(names_sorted + ["Net (sum + intercept)"])
    ax3.set_xlabel("Feature effect (left = toward Healthy, right = toward Tipburn)")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Explicit arithmetic, all numbers pulled from decomposition (nothing hardcoded).
    terms = [
        f"{feature_order[i]} ({decomposition.contributions[i]:+.2f})"
        for i in np.argsort(decomposition.contributions)[::-1]
    ]
    terms.append(f"intercept ({decomposition.intercept:+.2f})")
    arithmetic = " + ".join(terms) + f" = {decomposition.logit:+.2f}"
    st.info(
        f"Combined model value: {decomposition.logit:+.2f}  |  "
        f"Model score: {leaf_score:.3f}  |  "
        f"Classification: {predicted_class}"
    )

    st.caption(
        "The combined model value is the sum of the five feature effects and "
        "the intercept. The logistic function converts that value into the "
        "model score used for classification."
    )

    with st.expander("View calculation details"):
        calculation_html = (
            f"{arithmetic}<br>"
            f"Model score = {leaf_score:.3f}<br>"
            f"{leaf_score:.3f} is {relation} "
            f"{threshold} &rarr; {predicted_class}"
        )

        st.markdown(
            (
                '<div class="calculation-box">'
                f"{calculation_html}"
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        st.caption(
            (
                "Each feature effect equals its standardised "
                "feature value multiplied by the corresponding "
                "Logistic Regression coefficient."
            )
        )


def _clear_raw_extraction_state() -> None:
    for key in (
        "raw_prep_extracted",
        "raw_prep_proceed_confirmed",
        "raw_results",
        "raw_filter",
        "raw_results_select",
    ):
        st.session_state.pop(key, None)


def _reset_raw_prep_downstream_state() -> None:
    for key in (
        "raw_prep_ranges",
        "raw_prep_original_ranges",
    ):
        st.session_state.pop(key, None)

    _clear_raw_extraction_state()

    st.session_state[
        "raw_prep_table_version"
    ] = (
        st.session_state.get(
            "raw_prep_table_version",
            0,
        )
        + 1
    )


def _bump_table_version() -> None:
    st.session_state["raw_prep_table_version"] = st.session_state.get("raw_prep_table_version", 0) + 1


def render_prepare_raw_data_workflow(pipeline, metadata) -> None:
    st.write(
        "Upload a raw scan-level CSV, assign each scan range to its lettuce, leaf, and area, "
        "then continue to classification."
    )

    raw_upload = st.file_uploader(
        "Upload raw CSV",
        type=["csv"],
        key="raw_uploader",
        label_visibility="collapsed",
    )
    if raw_upload is None:
        _reset_raw_prep_downstream_state()
        st.info("Upload a raw CSV to begin labelling.")
        return

    raw_bytes = raw_upload.getvalue()
    if st.session_state.get("_raw_uploaded_file_name") != raw_upload.name:
        _reset_raw_prep_downstream_state()
        st.session_state["_raw_uploaded_file_name"] = raw_upload.name

    try:
        raw_source_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return

    st.write(
    f"Uploaded file: **{raw_upload.name}**. "
    f"**{len(raw_source_df)} rows**."
    )
    st.dataframe(raw_source_df.head(10))

    problems = validate_raw_upload_columns(raw_source_df)
    if problems:
        for problem in problems:
            st.error(problem)
        return

    st.subheader("1. Generate ranges")
    col1, col2, col3 = st.columns(3)
    start = col1.number_input(
        "First data-point row number", min_value=1, value=1, step=1, key="raw_prep_start",
        help="The row number (1-based) of the very first scan to include.",
    )
    gap = col2.number_input(
        "Number of scan rows per area", min_value=1, value=DEFAULT_GAP, step=1, key="raw_prep_gap",
        help="Number of scan rows per (Lettuce, Leaf, Area) combination.",
    )
    num_lettuces = col3.number_input(
        "Number of lettuces", min_value=1, value=1, step=1, key="raw_prep_num_lettuces",
        help="Generates three leaves and three areas for each lettuce.",
    )
    if st.button(
    "Generate ranges",
    type="primary",
    key="raw_prep_generate_manual",
    ):
        try:
            ranges = generate_manual_ranges(int(start), int(gap), int(num_lettuces))
        except RawDataPrepError as exc:
            st.error(str(exc))
        else:
            _clear_raw_extraction_state()
            st.session_state["raw_prep_ranges"] = ranges
            st.session_state["raw_prep_original_ranges"] = [
                item.copy() for item in ranges
            ]
            _bump_table_version()

    if "raw_prep_ranges" not in st.session_state:
        return

    st.subheader("2. Review and edit ranges")

    ranges = st.session_state[
        "raw_prep_ranges"
    ]

    if (
        "raw_prep_original_ranges"
        not in st.session_state
    ):
        st.session_state[
            "raw_prep_original_ranges"
        ] = [
            item.copy() for item in ranges
        ]

    original_ranges = st.session_state[
        "raw_prep_original_ranges"
    ]

    total_rows = len(raw_source_df)
    max_lettuce = max(
        item["lettuce"] for item in ranges
    )

    edit_mode_label = st.radio(
        "After editing",
        [
            "Adjust only the next range",
            "Shift all following ranges",
        ],
        horizontal=True,
        key="raw_prep_edit_mode_choice",
        help=(
            "Adjust only the next range changes the "
            "following start row. Shift all following "
            "ranges moves every later range while "
            "keeping its length."
        ),
    )

    edit_mode = (
        EDIT_MODE_ORIGINAL
        if edit_mode_label.startswith("Adjust")
        else EDIT_MODE_SHIFT_FOLLOWING
    )

    lettuce_key = (
        "raw_prep_selected_lettuce"
    )
    leaf_key = "raw_prep_selected_leaf"
    area_key = "raw_prep_selected_area"

    st.session_state[lettuce_key] = min(
        max(
            int(
                st.session_state.get(
                    lettuce_key,
                    1,
                )
            ),
            1,
        ),
        max_lettuce,
    )

    st.session_state[leaf_key] = min(
        max(
            int(
                st.session_state.get(
                    leaf_key,
                    1,
                )
            ),
            1,
        ),
        3,
    )

    st.session_state[area_key] = min(
        max(
            int(
                st.session_state.get(
                    area_key,
                    1,
                )
            ),
            1,
        ),
        3,
    )

    selection_col1, selection_col2, selection_col3 = (
        st.columns(3)
    )

    selected_lettuce = int(
        selection_col1.selectbox(
            "Lettuce number",
            options=list(range(1, max_lettuce + 1)),
            key=lettuce_key,
        )
    )

    selected_leaf = int(
        selection_col2.selectbox(
            "Leaf number",
            options=[1, 2, 3],
            key=leaf_key,
        )
    )

    selected_area = int(
        selection_col3.selectbox(
            "Area number",
            options=[1, 2, 3],
            key=area_key,
        )
    )

    selected_index = next(
        idx
        for idx, item in enumerate(ranges)
        if (
            item["lettuce"]
            == selected_lettuce
            and item["leaf"]
            == selected_leaf
            and item["area"]
            == selected_area
        )
    )

    selected_range = ranges[selected_index]

    editor_version = st.session_state.get(
        "raw_prep_table_version",
        0,
    )

    with st.form(
        "raw_prep_single_range_form"
    ):
        start_col, end_col = st.columns(2)

        edited_start = start_col.selectbox(
            "Start row",
            options=list(range(1, total_rows + 1)),
            index=int(selected_range["start"]) - 1,
            key=(
                f"raw_range_start_"
                f"{editor_version}_"
                f"{selected_index}"
            ),
        )

        edited_end = end_col.selectbox(
            "End row",
            options=list(range(1, total_rows + 1)),
            index=int(selected_range["end"]) - 1,
            key=(
                f"raw_range_end_"
                f"{editor_version}_"
                f"{selected_index}"
            ),
        )

        apply_edit = st.form_submit_button(
            "Apply changes",
            type="primary",
        )

    if apply_edit:
        ranges[selected_index]["start"] = int(
            edited_start
        )
        ranges[selected_index]["end"] = int(
            edited_end
        )

        apply_range_edit(
            ranges,
            selected_index,
            edit_mode,
        )

        st.session_state[
            "raw_prep_ranges"
        ] = ranges

        _clear_raw_extraction_state()
        _bump_table_version()
        st.rerun()

    with st.expander("View all ranges"):
        table_df = pd.DataFrame(
            ranges
        )[
            [
                "lettuce",
                "leaf",
                "area",
                "start",
                "end",
            ]
        ].rename(
            columns={
                "lettuce": "Lettuce",
                "leaf": "Leaf",
                "area": "Area",
                "start": "Start row",
                "end": "End row",
            }
        )

        editor_key = (
            f"raw_prep_table_editor_v"
            f"{editor_version}"
        )

        edited_df = st.data_editor(
            table_df,
            key=editor_key,
            num_rows="fixed",
            disabled=[
                "Lettuce",
                "Leaf",
                "Area",
            ],
            use_container_width=True,
            hide_index=True,
        )

        changed_index = None

        for idx in range(len(ranges)):
            if (
                int(
                    edited_df.iloc[idx][
                        "Start row"
                    ]
                )
                != ranges[idx]["start"]
                or int(
                    edited_df.iloc[idx][
                        "End row"
                    ]
                )
                != ranges[idx]["end"]
            ):
                changed_index = idx
                break

        if changed_index is not None:
            ranges[
                changed_index
            ]["start"] = int(
                edited_df.iloc[
                    changed_index
                ]["Start row"]
            )

            ranges[
                changed_index
            ]["end"] = int(
                edited_df.iloc[
                    changed_index
                ]["End row"]
            )

            apply_range_edit(
                ranges,
                changed_index,
                edit_mode,
            )

            st.session_state[
                "raw_prep_ranges"
            ] = ranges

            _clear_raw_extraction_state()
            _bump_table_version()
            st.rerun()

    modified_count = sum(
        1
        for current, original in zip(
            ranges,
            original_ranges,
        )
        if (
            current["start"]
            != original["start"]
            or current["end"]
            != original["end"]
        )
    )

    reset_col, status_col = st.columns(
        [1, 3]
    )

    with reset_col:
        if st.button(
            "Reset all ranges",
            use_container_width=True,
            key="raw_prep_reset_ranges",
        ):
            st.session_state[
                "raw_prep_ranges"
            ] = [
                item.copy()
                for item in original_ranges
            ]

            _clear_raw_extraction_state()
            _bump_table_version()
            st.rerun()

    with status_col:
        st.caption(
            f"{modified_count} range(s) differ "
            "from the generated values."
        )
    problems = validate_ranges(ranges, total_rows=total_rows)
    if problems:
        for problem in problems:
            st.error(problem)
        st.info("Fix the ranges above before confirming.")
        return

    covered_rows = sum(item["end"] - item["start"] + 1 for item in ranges)
    uncovered_intervals = find_uncovered_row_intervals(ranges, total_rows)
    if uncovered_intervals:
        interval_text = ", ".join(
            str(start_row)
            if start_row == end_row
            else f"{start_row}-{end_row}"
            for start_row, end_row
            in uncovered_intervals
        )

        uncovered_count = sum(
            end_row - start_row + 1
            for start_row, end_row
            in uncovered_intervals
        )

        st.warning(
            f"Unassigned raw-row interval(s): "
            f"{interval_text}. "
            f"{uncovered_count} row(s) will be excluded "
            "from extraction."
        )
    st.caption(f"{covered_rows} of {total_rows} raw rows are covered by the {len(ranges)} range(s) above.")

    if st.button(
        "Confirm and extract",
        type="primary",
        key="raw_prep_confirm",
    ):
        try:
            prepared_df = (
                extract_prepared_dataframe(
                    raw_source_df,
                    ranges,
                )
            )

        except RawDataPrepError as exc:
            st.error(str(exc))

        else:
            _clear_raw_extraction_state()

            st.session_state[
                "raw_prep_extracted"
            ] = prepared_df

            st.success(
                f"Extracted {len(prepared_df)} rows "
                f"across {len(ranges)} ranges."
            )

    if (
        "raw_prep_extracted"
        not in st.session_state
    ):
        return

    prepared_df = st.session_state[
        "raw_prep_extracted"
    ]

    with st.expander(
        "Preview prepared data"
    ):
        st.dataframe(
            prepared_df.head(20),
            use_container_width=True,
            hide_index=True,
        )

    edit_col, proceed_col, download_col = (
        st.columns([1, 1.3, 1.3])
    )

    with edit_col:
        if st.button(
            "Edit ranges",
            use_container_width=True,
            key="raw_prep_edit_ranges",
        ):
            _clear_raw_extraction_state()
            st.rerun()

    with proceed_col:
        if st.button(
            "Proceed to classification",
            type="primary",
            use_container_width=True,
            key="raw_prep_proceed",
        ):
            st.session_state[
                "raw_prep_proceed_confirmed"
            ] = True

    with download_col:
        st.download_button(
            "Download prepared CSV",
            data=prepared_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                "prepared_lettuce_data.csv"
            ),
            mime="text/csv",
            use_container_width=True,
            key="raw_prep_download",
        )

    if st.session_state.get(
        "raw_prep_proceed_confirmed"
    ):
        st.divider()
        st.subheader("Classification")

        render_classification_workflow(
            pipeline,
            metadata,
            prepared_df,
        )


def main() -> None:
    inject_dashboard_styles()
    render_title_banner()

    pipeline, metadata = get_pipeline()
    render_info_panel(metadata)

    render_prepare_raw_data_workflow(
        pipeline,
        metadata,
    )
        

if __name__ == "__main__":
    main()
