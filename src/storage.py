from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import json
from loguru import logger

from .config import TRANSCRIPTS_DIR

@dataclass
class TranscriptRecord:
    """
    Represents one saved transcript + metadata for later analysis
    """

    video_id: Optional[str]
    source: str         # e.g. "youtube", "twitter"
    url: str            # original video URL 
    title: Optional[str]    #
    transcript_text: str    # main text body
    transcript_raw: Dict[str, Any]  # full AssemblyAI JSON
    download_path: str      # where the audio file lives locally
    created_at: str         # ISO 8601 timestamp
    metrics: Dict[str, Any]  # view/day, like_rate, etc   

def create_transcript_record(
        *,
        source: str,
        url: str,
        download_path: Path,
        transcript_result: Dict[str, Any],
        video_id: Optional[str] = None,
        title: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
) -> TranscriptRecord:
    """
    Helper to build a TrascriptRecord from AssemblyAi result
    """
    text = transcript_result.get("text", "") or ""

    record = TranscriptRecord(
        video_id=video_id,
        source=source,
        url=url,
        title=title,
        transcript_text=text,
        transcript_raw=transcript_result,
        download_path=str(download_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        metrics=metrics or {},
    )

    return record

def _build_filename(record: TranscriptRecord) -> Path:
    """
    Decide how to name JOSN file on disk
    """
    ts = record.created_at.replace(":", "-")
    vid_part = record.video_id or "unknown_id"

    filename = f"{record.source}_{vid_part}_{ts}.json"
    return TRANSCRIPTS_DIR/filename

def save_transcript_record(record: TranscriptRecord) -> Path:
    """
    Serialize a TranscriptRecord to a JSON file under data/transcripts/ .
    :return Path to the saved JSON file
    """

    output_path = _build_filename(record)
    logger.info(f"Saving transcript record to {output_path}")

    data = asdict(record)

    # Make sure parant dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Transcript record saved.")
    return output_path

def load_transcript_record(path: Path) -> TranscriptRecord:
    """
    Load trascript record from JSON file
    """
    logger.info(f"Loading trascript record from {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    record = TranscriptRecord(**data)
    return record