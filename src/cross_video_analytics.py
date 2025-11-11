from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import pandas as pd
from loguru import logger

from .config import ANALYSIS_DIR, METRICS_DIR, DEFAULT_METRICS_CSV

# --------- Metricts loading and RPI compute ------

def load_metrics(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load YouTube video metritcs CSV

    Expected columns in the CSV (from your FullLenVideosCleaned file):
        - videoId
        - channelId
        - viewsPerDay
        - likeViewRatio
        - commentViewRatio
        - videoAgeInDays
        - Relative Performance Index (RPI) [optional: we can recompute if missing]
    """

    csv_path = csv_path or DEFAULT_METRICS_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metrics CSV not found at {csv_path}."
            f"Place your csv file there or pass a custom path"
        )
    
    logger.info(f"Loading metrics from: {csv_path}")
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names to avoid 'videoUrl ' vs 'videoUrl' issues
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    
    required_cols = [
        "videoId",
        "channelId",
        "viewsPerDay",
        "likeViewRatio",
        "commentViewRatio",
        "videoAgeInDays"
    ]

    missing = [ c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Metrics CSV is missing required columns: {missing}."
            f"Available: {list(df.columns)}"
        )
    
    return df

def compute_relative_performance_index(
        df: pd.DataFrame,
        max_age_days: int = 365,
) -> pd.DataFrame:
    """ 
    Compute Relative Performance Index (RPI)
    For each video (within_max_age_days):
    RPI = (VPD_video / VPD_avg_channel)
        + (LikeRate_video / LikeRate_avg_channel)
        + (CommentRate_video / CommentRate_avg_chennel)

    Where:
      - VPD = viewsPerDay
      - LikeRate = likeViewRatio
      - CommentRate = commentViewRatio

    The function returns a *filtered* DataFrame (only videos within max_age_days)
    with a new column 'relative_performance_index'.    
    """

    logger.info(f"Filtering videos to last {max_age_days} days by videoAgeInDays...")
    df_recent = df[df["videoAgeInDays"] <= max_age_days].copy()

    # Compute per-channel means
    logger.info("Computing channel-level averages (viewsPerDay, likeViewRatio, commentViewRatio)...")
    channel_means = (
        df_recent.groupby("channelId")
        .agg(
            avg_viewsPerDay=("viewsPerDay", "mean"),
            avg_likeViewRatio=("likeViewRatio", "mean"),
            avg_commentViewRatio=("commentViewRatio", "mean"),
        ).reset_index()
    )

    df_recent = df_recent.merge(channel_means, on="channelId", how="left")

    # Avoid division by zero by replacing 0 with Nan (hadle fill later)
    for avg_col in ["avg_viewsPerDay", "avg_likeViewRatio", "avg_commentViewRatio"]:
        df_recent.loc[df_recent[avg_col] == 0, avg_col] = pd.NA

    df_recent["rel_VPD"] = df_recent["viewsPerDay"] / df_recent["avg_viewsPerDay"]
    df_recent["rel_LikeRate"] = df_recent["likeViewRatio"] / df_recent["avg_likeViewRatio"]
    df_recent["rel_CommentRate"] = df_recent["commentViewRatio"] / df_recent["avg_commentViewRatio"]
    # df_recent["rel_CommentRate"] = df_recent["commentViewRatio"] / df_recent["avg_commentViewRatio"]

    # The sum of the three rations (Relative Performance Index)
    df_recent["relative_performance_index"] = (
        df_recent["rel_VPD"].fillna(0)
        + df_recent["rel_LikeRate"].fillna(0)
        + df_recent["rel_CommentRate"].fillna(0)
    )

    logger.info("RelativePerformance Index computed for recent videos.")
    return df_recent

def add_performance_metric_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a canonical 'performance_metric' column.
    Perference :
        1) Use existing 'Relative Performance Index (RPI)' column from your sheet.
        2) If not present, compute 'relative_performance_index' with the formula above.
    """

    rpi_col = "Relative Performance Index (PRI)"

    if rpi_col in df.columns:
        logger.info(f"using existing '{rpi_col}' as performance metric")
        df = df[df["videoAgeInDays"] <= 365].copy() # enforce 12-month filter
        df["performance_metric"] = df[rpi_col].astype(float)
        return df
    
    logger.info(f"'{rpi_col}' column not found; computing performance index from raw metrics... ")
    df_recent = compute_relative_performance_index(df)
    df_recent["performance_metric"] = df_recent["relative_performance_index"]
    return df_recent

# ----- LLM analysis loading ------------
def load_llm_analyses(analysis_dir: Optional[Path] = None) -> pd.DataFrame:
    """
        Load all *_llm_analysis_*.json files and flatten useful fields into a DataFrame
    """

    analysis_dir = analysis_dir or ANALYSIS_DIR
    pattern = "*_llm_analysis_*.json"
    files = list(analysis_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No LLM analysis files found in {analysis_dir} matching pattern {pattern}"
        )
    
    logger.info(f"Found {len(files)} LLM analysis files in {analysis_dir}")

    rows: list[Dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("_meta", {}) or {}
        hook = data.get("hook", {}) or {}
        psychology = data.get("psycology", {}) or {}
        style = data.get("style_and_tone", {}) or {}
        ctas = data.get("ctas", {}) or {}
        open_loops = data.get("open_loops", {}) or {}
        improvement = data.get("improvement_opportunities", {}) or {}

        row: Dict[str, Any] = {
                        "video_id": meta.get("video_id"),
            "source": meta.get("source"),
            "url": meta.get("url"),
            "title": meta.get("title"),
            "analyzed_at": meta.get("analyzed_at"),
            "llm_model": meta.get("model"),

            # Hook
            "hook_text": hook.get("hook_text"),
            "hook_type": hook.get("hook_type"),
            "hook_strength": hook.get("strength"),
            "hook_why_it_works": hook.get("why_it_works"),

            # Counts / simple aggregates
            "open_loops_count": len(open_loops),
            "primary_ctas_count": len(ctas.get("primary_ctas", []) or []),

                  # Psychology
            "target_audience": psychology.get("target_audience"),
            "core_mechanism": psychology.get("core_mechanism"),
            "pain_points_count": len(psychology.get("main_pain_points", []) or []),
            "desires_count": len(psychology.get("main_desires", []) or []),
            "objections_count": len(psychology.get("main_objections_addressed", []) or []),

            # Tone & style
            "overall_tone": style.get("overall_tone"),
            "language_style": style.get("language_style"),
            "pattern_interrupts_count": len(style.get("pattern_interrupts", []) or []),

            # Improvement
            "improve_hook_count": len(improvement.get("hook", []) or []),
            "improve_offer_cta_count": len(improvement.get("offer_and_cta", []) or []),
        }

        rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"LLM analysis DataFrame shape: {df.shape}")
    return df

# ----- Bucketing and summaries ------
def add_performance_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'performance bucket' column: 'low', 'mid', 'high' based on
    quantiles of 'performance_metric'. Also return top_20 and bottom_20 
    on that metric
    """
    metric = df["performance_metric"].astype(float)

    q_low = metric.quantile(0.2)
    q_high = metric.quantile(0.8)

    def label(value: float) -> str:
        if value <= q_low:
            return "low"
        if value >= q_high:
            return "high"
        return "mid"
    
    logger.info(
        f"Performance buckets labeled using 'performance_metric' "
        f"(low <= {q_low:.2f}, high >= {q_high:.2f})"
    )

    return df

def build_summary_tables(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build aggregation tables for high/mid/low vs. hook type, tone, etc
    """

    tables: dict[str, pd.DataFrame] = {}

    # Hook type vs performance
    if {"hook_type", "performance_bucket"}.issubset(combined.columns):
        hook_perf = (
            combined.groupby(["hook_type", "performance_bucket"])
            .agg(
                count=("video_id", "count"),
                avg_metric=("performance_metric", "mean"),
            )
            .reset_index()
        )
        tables["hook_vs_performance"] = hook_perf

    # Hook strenght vs performance
    if {"hook_strength", "performance_bucket"}.issubset(combined.columns):
        hook_strength_perf = (
            combined.groupby(["hook_strength", "performance_bucket"])
            .agg(
                count=("video_id", "count"),
                avg_metric=("performance_metric", "metric"),
            )
            .reset_index()
        )
        tables["hook_strength_vs_performance"] = hook_strength_perf

    # Tone vs performance
    if {"overall_tone", "performance_bucket"}.issubset(combined.columns):
        tone_perf = (
            combined.groupby(["overall_tone", "performance_bucket"])
            .agg(
                count=("video_id", "count"),
                avg_metric=("performance_metric", "mean"),
            )
            .reset_index()
        )
        tables["tone_vs_performance"] = tone_perf

    # Pattern interrupts count vs performance
    if "perttern_interrupts_count" in combined.columns:
        tmp = combined.copy()
        tmp["pattern_interrupts_bucket"] = pd.cut(
            tmp["pattern_interrupts_count"],
            bins=[-1, 0, 1, 3, 100],
            labels=["0", "1", "2-3", "4+"]
        )

        patt_perf = (
            tmp.groupby(["pattern_interrupts_bucket", "performance_bucket"])
            .agg(
                count=("video_id", "count"),
                avg_metric=("performance_metric", "mean")
            )
            .reset_index()
        )
        tables["pattern_interrupts_vs_performance"] = patt_perf

    return tables

# ---------- Main orchestration ----------
def run_cross_video_analytics(
        metrics_csv: Optional[Path] = None,
        output_dir: Optional[Path] = None
) -> None:
    """
    End-to-end cross-video analytics:

    - Load metrics CSV (FullLenVideosCleaned).
    - Restrict to last 12 months.
    - Use RPI (or compute a similar index) as performance_metric.
    - Load LLM per-video analysis.
    - Join on video_id.
    - Label videos as low/mid/high.
    - Export combined data & summary tables to CSV.
    """

    output_dir = output_dir or METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metics + performance metric
    metrics_df = load_metrics(metrics_csv)
    metrics_df = add_performance_metric_column(metrics_df)

    # Normalize video id column name
    metrics_df = metrics_df.rename(columns={"videoId": "video_id"})

    #2. LLM analysis
    llm_df = load_llm_analyses()

    #3. Inner join: keep only videos that jave both metrics + LLM analysis
    combined = pd.merge(
        llm_df,
        metrics_df,
        on="video_id",
        how="inner",
        suffixes=("_llm", "_metrics"),
    )

    logger.info(f"Combined DataFrame shape after join: {combined.shape}")

    #4. Bucket performance
    combined = add_performance_buckets(combined)

    #5 Save combined dataset
    combined_csv_path = output_dir / "combined_analytics.csv"
    combined.to_csv(combined_csv_path, index=False)
    logger.info(f"Combined analytics CSV saved to: {combined_csv_path}")

    #6.  Top / bottom 20 (by performance_metric) for quick inspection
    combined_sorted = combined.sort_values("performance_metric", ascending=False)
    top_20 = combined_sorted.head(20)
    bottom_20 = combined_sorted.tail(20)

    top_20.to_csv(output_dir / "top_20_videos.csv", index=False)
    bottom_20.to_csv(output_dir/"bottom_20_videos.csv", index=False)

    logger.info(f"Top 20 videos saved to: {output_dir/'top_20_videos.csv'}")
    logger.info(f"Botton 20 video saved to: {output_dir / 'bottom_20_videos.csv'}")

    #7 Summary table
    tables = build_summary_tables(combined)
    for name, df_tbl in tables.items():
        path = output_dir / f"summary_{name}.csv"
        df_tbl.to_csv(path, index=False)
        logger.info(f"Summary '{name}' saved to: {path}")