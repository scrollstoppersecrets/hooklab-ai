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

    rpi_col = "Relative Performance Index (RPI)"

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
    including hooks, loops, CTAs, psychology, and style/tone
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

        # Open loops
        loops_list = open_loops if isinstance(open_loops, list) else []
        loops_count = len(loops_list)
        unresolved_count = sum(1 for l in loops_list if not l.get("is_resolved", False))
        where_values = [l.get("where_resolved") for l in loops_list if l.get("where_resolved")]
        loops_mode = None
        if where_values:
            s = pd.Series(where_values)
            loops_mode = s.mode().iloc[0] if not s.mode().empty else None
        loops_sample = " | ".join([l.get("text", "") for l in loops_list[:2] if l.get("text")])

        # CTAs
        primary_ctas = ctas.get("primary_ctas") or []
        primary_ctas = primary_ctas if isinstance(primary_ctas, list) else []
        cta_count = len(primary_ctas)
        cta_types = sorted({(c.get("cta_type") or "").strip() for c in primary_ctas if c.get("cta_type")})
        cta_types_joined = ";".join(cta_types) if cta_types else None

        def _pos_count(pos: str) -> int:
            return sum(1 for c in primary_ctas if (c.get("position") or "").lower() == pos)
        
        cta_pos_early = _pos_count("early")
        cta_pos_middle = _pos_count("middle")
        cta_pos_late = _pos_count("late")

        # Psychology
        pains = psychology.get("main_pain_points") or []
        desires = psychology.get("main_desires") or []
        objections = psychology.get("main_objections_addrressed") or []
        pains_top3 = "; ".join(pains[:3]) if pains else None
        desires_top3 = "; ".join(desires[:3]) if desires else None
        objections_top3 = "; ".join(objections[:3]) if objections else None
        credibitily = psychology.get("credibility_elements") or []
        credibility_count = len(credibitily)
        credibility_sample = credibitily[0] if credibitily else None

        # Style & tone
        pattern_interrupts = style.get("pattern_interrupts") or []
        pattern_interrupts_count = len(pattern_interrupts)
        memmorable_lines = style.get("memorable_lines") or []
        memmorable_line_sample = memmorable_lines[0] if memmorable_lines else None

        row: Dict[str, Any] = {
            # meta
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

            # Open loops
            # "open_loops_count": len(open_loops),
            "open_loops_count": loops_count,
            "unresolved_loops_count": unresolved_count,
            "loops_where_resolved_mode": loops_mode,
            "open_loops_sample": loops_sample,

            # CTAs
            "primary_ctas_count": cta_count,
            "cta_types": cta_types_joined,
            "cta_pos_early": cta_pos_early,
            "cta_pos_middle": cta_pos_middle,
            "cta_pos_late": cta_pos_late,

            # Psychology
            "target_audience": psychology.get("target_audience"),
            "core_mechanism": psychology.get("core_mechanism"),
            "pain_points_count": len(psychology.get("main_pain_points", []) or []),
            "desires_count": len(psychology.get("main_desires", []) or []),
            "objections_count": len(psychology.get("main_objections_addressed", []) or []),
            "pain_points_top3": pains_top3,
            "desires_top3": desires_top3,
            "objections_top3": objections_top3,
            "credibility_count": credibility_count,
            "credibility_sample": credibility_sample,

            # Tone & style
            "overall_tone": style.get("overall_tone"),
            "language_style": style.get("language_style"),
            "pattern_interrupts_count": len(style.get("pattern_interrupts", []) or []),
            "memorable_lines_count": memmorable_lines,
            "memorable_line_sample": memmorable_line_sample,

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

    df = df.copy()
    df["performance_bucket"] = metric.apply(label)
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
                avg_metric=("performance_metric", "mean"),
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
    if "pattern_interrupts_count" in combined.columns:
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
    # Handle smaller dataset sizes
    n = len(combined_sorted)
    if n == 0:
        logger.warning("No rows in combined dataset; skipping top/bottom exports.")
        return
    
    k = min(20, max(1, n // 2)) # <= half the set, at least 1
    top_k = combined_sorted.head(k)
    bottom_k = combined_sorted.tail(k)

    top_performers_file_name = f"top_{k}_videos.csv"
    bottom_performers_file_name = f"bottom_{k}_videos.csv"

    path_to_top_performers_file = output_dir / top_performers_file_name
    path_to_bottom_performers_file = output_dir / bottom_performers_file_name

    top_k.to_csv(path_to_top_performers_file, index=False)
    bottom_k.to_csv(path_to_bottom_performers_file, index=False)

    logger.info(f"Top {k} videos saved to: {path_to_top_performers_file}")
    logger.info(f"Bottom {k} videos saved to: {path_to_bottom_performers_file}")

    # Always write fixed top/bottom 20 when dataset is large enough
    if n >= 40:
        top20_path = output_dir / "top_20_videos.csv"
        bottom20_path = output_dir / "bottom_20_videos.csv"
        combined_sorted.head(20).to_csv(top20_path, index=False)
        combined_sorted.tail(20).to_csv(bottom20_path, index=False)
        logger.info(f"Top 20 videos saved to: {top20_path}")
        logger.info(f"Bottom 20 videos saved to: {bottom20_path}")
    # Top 20/Bottom 20 results of the larger datasets
    # top_20 = combined_sorted.head(20)
    # bottom_20 = combined_sorted.tail(20)

    # top_20.to_csv(output_dir / "top_20_videos.csv", index=False)
    # bottom_20.to_csv(output_dir/"bottom_20_videos.csv", index=False)

    # logger.info(f"Top 20 videos saved to: {output_dir/'top_20_videos.csv'}")
    # logger.info(f"Botton 20 video saved to: {output_dir / 'bottom_20_videos.csv'}")

    #7 Summary table
    tables = build_summary_tables(combined)
    for name, df_tbl in tables.items():
        path = output_dir / f"summary_{name}.csv"
        df_tbl.to_csv(path, index=False)
        logger.info(f"Summary '{name}' saved to: {path}")