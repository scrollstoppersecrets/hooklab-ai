from __future__ import annotations
import sys
import argparse
from loguru import logger
from .pipeline_batch import run_batch
from .cross_video_analytics import run_cross_video_analytics

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run HookLab Ai end-to-end: batch transcription -> cross-video analytics."
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument("--batch", type=int, default=None,
                   help="Process N videos from metrics CSV (then run analytics).")
    g.add_argument("--all", action="store_true",
                   help="Process ALL videos from metrics CSV (then run analytics).")
    g.add_argument("--skip-batch", action="store_true",
                   help="Skip transcription batch, only run analytics.")
    g.add_argument("--only-batch", action="store_true",
                   help="Run transcription batch only (no analytics).")
    g.add_argument("--recent-days", type=int, default=None,
                   help="(Optional) Pass-through filter for batch loader (e.g., 365).")
    g.add_argument("--quiet", action="store_true",
                   help="Less verbose logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # Decide batch size (None mean: ALL (runs the whole 'metrics/videos-data' file))
    batch_size = None
    if args.batch is not None:
        batch_size = args.batch
    elif args.all:
        batch_size = None # explicit clarity: process all

    # Phase A: Transcription batch (unless skipped)
    if not args.skip_batch:
        logger.info("Phase A: Trasncription/LLM batch starting...")
        try:
            """
            run_batch reads the metrics CSV, downloads, transcribes, 
            saves JSON, creates Top20/Bottom20 csv files
            """ 
            run_batch(batch_size=batch_size)
            logger.info("Phase A - completed.")
        except Exception as exc:
            logger.error(f"Phase A failed: {exc}")
            # Keep going only if the user asked for analytics-only; 
            # otherwise exit non-zero
    
    if args.only_batch:
        logger.info("Only-batch flag set; skipping Phase B analytics.")
        return
    
    # Phase B: Cross-video analytics
    logger.info("Phase B - Cross-Video Analytics starting...")
    try:
        run_cross_video_analytics()
        logger.info("Phase B - completed.")
    except Exception as exc:
        logger.error(f"Phase B failed: {exc}")
        raise

if __name__ == "__main__":
    main()