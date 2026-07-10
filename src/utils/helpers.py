"""
Shared helper utilities for the Binance Spot K-lines pipeline.

Provides common timestamp conversions and markdown report formatting tools.
"""

import datetime
from pathlib import Path

import pandas as pd


def ms_to_str(ms: int) -> str:
    """Converts epoch milliseconds to formatted UTC timestamp string."""
    try:
        return datetime.datetime.fromtimestamp(ms / 1000.0, datetime.UTC).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (ValueError, OSError, OverflowError):
        return str(ms)


def generate_profile_markdown(df_profile: pd.DataFrame, title: str, description: str) -> str:
    """Calculates missing intervals and formats aggregation statistics into a markdown report.

    Assumes input DataFrame contains columns: symbol, row_count, min_time_ms,
    max_time_ms, duplicate_timestamps, null_values_count.
    """
    df_profile = df_profile.copy()
    df_profile["start_date"] = df_profile["min_time_ms"].apply(ms_to_str)
    df_profile["end_date"] = df_profile["max_time_ms"].apply(ms_to_str)

    # Calculate missing intervals (1 minute = 60,000 milliseconds)
    df_profile["expected_rows"] = (
        (df_profile["max_time_ms"] - df_profile["min_time_ms"]) / 60000 + 1
    ).astype(int)
    df_profile["missing_rows"] = df_profile["expected_rows"] - df_profile["row_count"]
    df_profile["gap_percentage"] = (
        df_profile["missing_rows"] / df_profile["expected_rows"] * 100
    ).round(4)

    # Generate summary metrics
    total_symbols = len(df_profile)
    total_rows = df_profile["row_count"].sum()
    total_duplicates = df_profile["duplicate_timestamps"].sum()
    total_nulls = df_profile["null_values_count"].sum()

    report_lines = [
        f"# {title}",
        "",
        description,
        "",
        "## Summary Metrics",
        "",
        f"- **Total Unique Symbols**: {total_symbols}",
        f"- **Total Rows (Observations)**: {total_rows:,}",
        f"- **Total Duplicate Timestamps**: {total_duplicates}",
        f"- **Total Rows with Null Values**: {total_nulls}",
        "",
        "## Detailed Symbol Analysis",
        "",
        "| Symbol | Total Rows | Expected Rows | Gaps (Rows) | Gap % | "
        "Start Date (UTC) | End Date (UTC) | Duplicates | Nulls |",
        "| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: |",
    ]

    report_lines.extend(
        f"| {row.symbol} | {row.row_count:,} | {row.expected_rows:,} | {row.missing_rows:,} | "
        f"{row.gap_percentage}% | {row.start_date} | {row.end_date} | "
        f"{row.duplicate_timestamps} | {row.null_values_count} |"
        for row in df_profile.itertuples(index=False)
    )

    return "\n".join(report_lines) + "\n"


def discover_symbol_csvs(symbols: list[str], base_dir: "Path") -> list["Path"]:
    """Discovers 1m kline CSV files for a specific list of symbols."""
    valid_paths = []
    for symbol in symbols:
        symbol_dir = base_dir / "spot" / "monthly" / "klines" / symbol / "1m"
        if symbol_dir.exists():
            csv_files = list(symbol_dir.glob("*.csv"))
            if csv_files:
                valid_paths.extend(csv_files)
    return valid_paths


def discover_all_csvs(base_dir: "Path") -> list["Path"]:
    """Discovers all 1m kline CSV files across all symbols."""
    return list((base_dir / "spot" / "monthly" / "klines").glob("*/1m/*.csv"))
