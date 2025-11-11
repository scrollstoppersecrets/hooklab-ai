from __future__ import annotations
from loguru import logger
from .cross_video_analytics import run_cross_video_analytics

def main() -> None:
    logger.info("Running Phase 4 - Cross-Video Analytics..")
    run_cross_video_analytics()
    logger.info("Done")

if __name__ == "__main__":
    main()