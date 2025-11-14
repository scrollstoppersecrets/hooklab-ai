from __future__ import annotations

from pathlib import Path
from typing import Optional

import random, time

import pandas as pd
from loguru import logger

from .config import (
    ANALYSIS_DIR, 
    MIN_DOWNLOAD_SLEEP_SECONDS, 
    MAX_DOWNLOAD_SLEEP_SECONDS, 
    MAX_VIDEOS_PER_RUN,
    LOGS_DIR,
)

from .cross_video_analytics import load_metrics
from .youtube_downloader import download_audio
from .transcriber import transcribe_file
from .storage import create_transcript_record, save_transcript_record
from .analysis import analyze_transcript_with_llm, save_llm_analysis

def _is_already_processed(video_id: str) -> bool:
    """
    Check if we've already run LLM analysis for this video_id
    If *_llm_analysis_*.json file with this ID exist, skip it
    """

    if not video_id: 
        return False
    
    pattern = f"*_{video_id}_llm_analysis_*.json"
    matches = list(ANALYSIS_DIR.glob(pattern))
    return len(matches) > 0

def _get_video_url(row: pd.Series) -> Optional[str]:
    """
    Decide how to get the video URL from a metric row.
    Prefer a 'videoUrl' column if it exists, fallback to building from videoId.
    """

    if "videoUrl" in row and isinstance(row["videoUrl"], str) and row["videoUrl"].strip():
        return row["videoUrl"].strip()
    
    video_id = str(row.get("videoId") or "").strip()
    if not video_id:
        return None
    
    return f"https://www.youtube.com/watch?v={video_id}"

def process_video_row(row: pd.Series) -> None:
    """
    Process a single video row from the metrics CSV:

    - Download audio from YouTube
    - Transcribe with AssemblyAI
    - Save TranscriptRecord JSON
    - Run LLM analysis and save *_llm_analysis_*.json
    - Delete local audio file
    """
    video_id = str(row.get("videoId") or "").strip()
    title = str(row.get("title") or "").strip()

    if not video_id:
        logger.info("Row has no videoId; skiping.")
        return 
    
    if _is_already_processed(video_id):
        logger.info(f"Video {video_id} already has LLM analysis; skipping.")
        return
    
    url = _get_video_url(row)
    if not url:
        logger.warning(f"Could not determine URL for video_id={video_id}");
        return
    
    logger.info(f"Processing video_id={video_id} title={title!r} url={url}")

    # 1. Download audio
    audio_path = download_audio(url)
    if not audio_path:
        logger.error(f"Download failed for vide_id={video_id}; skipping.")
        return

    try:
        # 2. Transcribe
        transcribe_results = transcribe_file(audio_path)

        # 3. Build and save transcript record
        record = create_transcript_record(
            source="youtube",
            url=url,
            download_path=audio_path,
            transcript_result=transcribe_results,
            video_id=video_id,
            title=title,
            # Placeholder for raw metrics metrics={}
            metrics={}
        )
        transcript_path = save_transcript_record(record)
        logger.info(f"Transcript saved to: {transcript_path}")

        # 4. LLM Analysis
        llm_analysis = analyze_transcript_with_llm(record)
        llm_path = save_llm_analysis(record, llm_analysis)
        logger.info(f"LLM analysis saved to: {llm_path}")

    except Exception as exc:
        logger.error(f"Error processing video_id={video_id}: {exc}")

    finally:
        # 5. Delete local audio to save space
        try:
            audio_path.unlink(missing_ok=True)
            logger.info(f"Deleted local audio file: {audio_path}")
        except Exception as exc:
            logger.warning(f"Failed to delete audio file {audio_path}: {exc}")

def run_batch(batch_size: Optional[int] = None) -> None:
    """
    Run the pipeline for a batch of videos from the metrics CSV.

    Strategy: 
    - Load all metrics
    - Filter to only videos that do NOT yet have LLM analysis (unprocessed).
    - Apply batch_size / MAX_VIDEOS_PER_RUN as a limit on that upprocessed subset.
    - Process those rows one by one with random sleep between them.

    - If batch_size is None: process all rows.
    - If batch_size is an int: process only that many rows (from the top).

    This does NOT alter the CSV; it just reads from it and writes transcripts / analyses.
    """

    metrics_df = load_metrics()
    # You can filter here if you only want to recent videos, e.g. <= 365 days
    # metrics_df = metrics_df[metrics_df["videoAgeInDays"] <= 365]


    # Build a filter DataFrame on only "unprocessed" videos
    def _row_unprocessed(row: pd.Series) -> bool:
        video_id = str(row.get("videoId") or "").strip()
        if not video_id:
            return False
        return not _is_already_processed(video_id)
    
    unprocessed_df = metrics_df[metrics_df.apply(_row_unprocessed, axis=1)]

    logger.info(
        f"Found {len(unprocessed_df)} unprocessed videos out of "
        f"{len(metrics_df)} total."
    )

    # Decide the effective limit for this run
    limit: Optional[int] = None

    # if batch_size is not None:
    #     logger.info(f"Running batch for first {batch_size} rows of metrics...")
    #     metrics_df = metrics_df.head(batch_size)
    # else:
    #     logger.info("Running batch for ALL rowa in metrics CSV...")
    if batch_size is not None and batch_size > 0:
        limit = batch_size

    if MAX_VIDEOS_PER_RUN and MAX_VIDEOS_PER_RUN > 0:
        limit = min(limit, MAX_VIDEOS_PER_RUN) if limit is not None else MAX_VIDEOS_PER_RUN

    if limit is not None:
        logger.info(f"Limiting this run to {limit} unprecessed video.")
        unprocessed_df = unprocessed_df.head(limit)
    else:
        logger.info("No per-run limit set; processing all unprocessed videos.")

    # Iterate row by row
    for idx, row in unprocessed_df.iterrows():
        logger.info(f"--- Processing row index={idx} videoId={row.get('videoId')} ----")
        process_video_row(row)

        # Random jitter between videos to avoid hammering tooo many requests
        delay = random.uniform(MIN_DOWNLOAD_SLEEP_SECONDS, MAX_DOWNLOAD_SLEEP_SECONDS)
        logger.info(f"Sleeping {delay: .1f}s belay next video ...")
        time.sleep(delay)

if __name__ == "__main__":
    import sys

    # Configure file logging for this run
    log_file = LOGS_DIR / "pipeline_batch_{time}.log"
    logger.add(
        log_file,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    logger.info(f"Logging to file: {log_file}")

    # Allow optional batch from CLI: python -m src.pipeline_batch [batch_size]
    if len(sys.argv) > 1:
        try:
            batch_size_arg = int(sys.argv[1])
        except ValueError:
            batch_size_arg = None
            print(f"Invalid batch argumant {sys.argv[1]!r}; defaulting to full file.")
    else:
        batch_size_arg = None

    run_batch(batch_size=batch_size_arg)