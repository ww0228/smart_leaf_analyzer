"""Prepare raw ColorCompass scan rows for dashboard classification.

The module generates, validates, edits, and extracts inclusive 1-based
row ranges for each lettuce, leaf, and measurement area.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.validation import detect_wavelength_columns  # noqa: E402

RAW_WAVELENGTH_MIN = 340
RAW_WAVELENGTH_MAX = 850
RAW_EXPECTED_BAND_COUNT = (RAW_WAVELENGTH_MAX - RAW_WAVELENGTH_MIN) // 5 + 1
LEAVES_PER_LETTUCE = 3
AREAS_PER_LEAF = 3
DEFAULT_GAP = 10

EDIT_MODE_ORIGINAL = "original"
EDIT_MODE_SHIFT_FOLLOWING = "shift-following"


class RawDataPrepError(ValueError):
    """Raised for user-facing validation problems in the raw-data-prep flow."""


def validate_raw_upload_columns(df: pd.DataFrame) -> list[str]:
    """Return human-readable problems with an uploaded raw CSV; empty list means it's usable."""
    problems: list[str] = []
    wavelength_cols = detect_wavelength_columns(
        df,
        wl_min=RAW_WAVELENGTH_MIN,
        wl_max=RAW_WAVELENGTH_MAX,
    )
    if len(wavelength_cols) < RAW_EXPECTED_BAND_COUNT:
        problems.append(
            f"Found only {len(wavelength_cols)} wavelength columns in the "
            f"{RAW_WAVELENGTH_MIN}-{RAW_WAVELENGTH_MAX} nm range (need {RAW_EXPECTED_BAND_COUNT}, "
            f"i.e. {RAW_WAVELENGTH_MIN}-{RAW_WAVELENGTH_MAX} nm in 5 nm steps). Check the file has "
            "one column per wavelength with numeric headers."
        )
    return problems


def generate_manual_ranges(start: int, gap: int, num_lettuces: int) -> list[dict[str, int]]:
    """Generate one inclusive row range for each lettuce, leaf, and area."""
    if gap < 1:
        raise RawDataPrepError("Gap size must be a positive integer.")
    if num_lettuces < 1:
        raise RawDataPrepError("Number of lettuces must be a positive integer.")
    if start < 1:
        raise RawDataPrepError("First data-point row number must be a positive integer.")

    ranges: list[dict[str, int]] = []
    current_start = start
    for lettuce in range(1, num_lettuces + 1):
        for leaf in range(1, LEAVES_PER_LETTUCE + 1):
            for area in range(1, AREAS_PER_LEAF + 1):
                start_num = current_start
                end_num = current_start + gap - 1
                ranges.append({
                    "lettuce": lettuce,
                    "leaf": leaf,
                    "area": area,
                    "start": start_num,
                    "end": end_num,
                })
                current_start = end_num + 1
    return ranges


def validate_ranges(ranges: list[dict[str, int]], total_rows: int) -> list[str]:
    """Return human-readable problems with a set of ranges; empty list means they're usable."""
    problems: list[str] = []
    for item in ranges:
        label = f"Lettuce {item['lettuce']} / Leaf {item['leaf']} / Area {item['area']}"
        if item["start"] > item["end"]:
            problems.append(f"{label}: start row ({item['start']}) is after end row ({item['end']}).")
        if item["start"] < 1 or item["end"] > total_rows:
            problems.append(
                f"{label}: range [{item['start']}, {item['end']}] falls outside the file's "
                f"1-{total_rows} row bounds."
            )

    by_start = sorted(ranges, key=lambda item: item["start"])
    for previous, current in zip(by_start, by_start[1:]):
        if current["start"] <= previous["end"]:
            problems.append(
                f"Overlapping ranges: [{previous['start']}, {previous['end']}] and "
                f"[{current['start']}, {current['end']}]."
            )
    return problems


def find_uncovered_row_intervals(
    ranges: list[dict[str, int]],
    total_rows: int,
) -> list[tuple[int, int]]:
    """Return raw-row intervals that are not assigned to any range."""
    if total_rows < 1:
        return []

    if not ranges:
        return [(1, total_rows)]

    ordered_ranges = sorted(ranges, key=lambda item: item["start"])
    uncovered: list[tuple[int, int]] = []
    next_uncovered_start = 1

    for item in ordered_ranges:
        range_start = int(item["start"])
        range_end = int(item["end"])

        if range_start > next_uncovered_start:
            uncovered.append(
                (next_uncovered_start, range_start - 1)
            )

        next_uncovered_start = max(
            next_uncovered_start,
            range_end + 1,
        )

    if next_uncovered_start <= total_rows:
        uncovered.append(
            (next_uncovered_start, total_rows)
        )

    return uncovered


def shift_following_ranges(ranges: list[dict[str, int]], edited_index: int) -> None:
    """Shift every range after edited_index so each keeps its original length, back-to-back."""
    next_start = ranges[edited_index]["end"] + 1
    for item in ranges[edited_index + 1:]:
        length = item["end"] - item["start"] + 1
        item["start"] = next_start
        item["end"] = item["start"] + length - 1
        next_start = item["end"] + 1


def adjust_next_range(ranges: list[dict[str, int]], edited_index: int) -> None:
    """Nudge only the immediately-following range's start to butt up against the edited range."""
    if edited_index + 1 >= len(ranges):
        return
    ranges[edited_index + 1]["start"] = ranges[edited_index]["end"] + 1


def apply_range_edit(ranges: list[dict[str, int]], edited_index: int, edit_mode: str) -> None:
    if edit_mode == EDIT_MODE_SHIFT_FOLLOWING:
        shift_following_ranges(ranges, edited_index)
    else:
        adjust_next_range(ranges, edited_index)


def extract_prepared_dataframe(raw_df: pd.DataFrame, ranges: list[dict[str, int]]) -> pd.DataFrame:
    """Slice raw_df by each range and prepend Lettuce_Sample_ID/Leaf_ID/Area_ID, matching the
    reference script's column order and iloc slicing convention exactly.
    """
    if not ranges:
        raise RawDataPrepError("No ranges to extract.")

    pieces = []
    for item in ranges:
        chunk = raw_df.iloc[item["start"] - 1: item["end"]].copy()
        chunk.insert(0, "Area_ID", item["area"])
        chunk.insert(0, "Leaf_ID", item["leaf"])
        chunk.insert(0, "Lettuce_Sample_ID", item["lettuce"])
        pieces.append(chunk)

    return pd.concat(pieces, axis=0, ignore_index=True)


__all__ = [
    "RawDataPrepError",
    "RAW_WAVELENGTH_MIN",
    "RAW_WAVELENGTH_MAX",
    "RAW_EXPECTED_BAND_COUNT",
    "LEAVES_PER_LETTUCE",
    "AREAS_PER_LEAF",
    "DEFAULT_GAP",
    "EDIT_MODE_ORIGINAL",
    "EDIT_MODE_SHIFT_FOLLOWING",
    "validate_raw_upload_columns",
    "generate_manual_ranges",
    "validate_ranges",
    "shift_following_ranges",
    "adjust_next_range",
    "apply_range_edit",
    "extract_prepared_dataframe",
    "find_uncovered_row_intervals",
]
