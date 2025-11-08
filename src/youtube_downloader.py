from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import yt_dlp
from loguru import logger

from .config import DOWNLOADS_DIR, YTDLP_OUTPUT_TEMPLATE, DEFAULT_AUDIO_FORMAT

def _build_yt_dlp_opts() -> Dict[str, Any]:
    """
    Building the options dictionary for yt-dlp
    """

    return {
        "format": DEFAULT_AUDIO_FORMAT,
        "outtmpl": YTDLP_OUTPUT_TEMPLATE,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True
    }

def download_audio(youtube_url: str) -> Optional[Path]:
    """
    Download the audio track from a YouTube video.
    """
    logger.info(f"Stating download for URL: {youtube_url}")
    ydl_opts = _build_yt_dlp_opts()

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # extract_info returns metadata about the video.
            # download=True tell yt-dlp to actually save the sile
            info = ydl.extract_info(youtube_url, download=True)
    except Exception as exc:
        logger.error(f"Error downloading {youtube_url}: {exc}")
        return None
    
    # Try to determine the final file path
    video_id = info.get("id")
    ext = info.get("ext", "webm") # yt-dlp should set this , but this is a fallback
    if not video_id:
        logger.error("yt-dlp did not return a video ID; cannot determine file path")
        return None
    
    expected_path = DOWNLOADS_DIR / f"{video_id}.{ext}"

    if expected_path.exists():
        logger.info(f"Download complete: {expected_path}")
        return expected_path
    
    # Fallback in case yt-dlp remuxes to anthothe extension (e.g., m4a/mp3)
    candidates = list(DOWNLOADS_DIR.glob(f"{video_id}.*"))
    if candidates: 
        actual_path = candidates[0]
        logger.warning(
            f"Expected file {expected_path} not found, but found {actual_path} instead."   
        )
        return actual_path
    
    logger.error(
        f"Download was reported as success, but no file found for video_id={video_id} "
        f"in {DOWNLOADS_DIR}"
    )
    return None