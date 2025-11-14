from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
from datetime import datetime, timezone

from loguru import logger
from openai import OpenAI

from .config import ANALYSIS_DIR, PROMPTS_DIR, get_openai_api_key
from .storage import TranscriptRecord

# -----  OpenAI client ----
_client = OpenAI(api_key=get_openai_api_key)

# ---- Prompt & schema ---- 
def _load_prompt(name: str) -> str:
  """
  Load a prompt text file from prompt dir
  """
  path = PROMPTS_DIR / name
  if not path.exists():
    raise FileNotFoundError(f"Prompt file not found: {path}")
  return path.read_text(encoding="utf-8")


_ANALYSIS_SYSTEM_PROMPT = _load_prompt("llm_analysis_system_prompt.txt")


# This describes the JSON shape we want the model to produce.
# We don't enforce a full JSON Schema in code, but we describe it clearly in the prompt.
_ANALYSIS_USER_INSTRUCTIONS = _load_prompt("llm_analysis_userm_prompt.txt")

def _build_user_content(record: TranscriptRecord) -> str:
    """
    Build the full user content for the LLM call, including metadata + transcript.
    """
    meta_lines = [
        f"Source: {record.source}",
        f"Video ID: {record.video_id}",
        f"URL: {record.url}",
        f"Title: {record.title}",
        "",
        "Full transcript:",
        record.transcript_text or "",
    ]
    return _ANALYSIS_USER_INSTRUCTIONS + "\n\n" + "\n".join(meta_lines)


def analyze_transcript_with_llm(
    record: TranscriptRecord,
    model: str = "gpt-5.1",   # or "gpt-4.1" / "o4-mini" depending on cost/quality
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Call OpenAI to get a structured JSON breakdown of a single video transcript.
    """
    logger.info(f"Starting LLM analysis for video_id={record.video_id} url={record.url}")

    user_content = _build_user_content(record)

    response = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_content},
        ],
    )

    content = response.choices[0].message.content
    try:
        analysis_dict = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse LLM JSON output: {exc}. Raw content: {content!r}")
        raise

    # Optionally enrich with metadata + timestamp
    analysis_dict.setdefault("_meta", {})
    analysis_dict["_meta"].update(
        {
            "video_id": record.video_id,
            "source": record.source,
            "url": record.url,
            "title": record.title,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
        }
    )

    logger.info("LLM analysis completed.")
    return analysis_dict


def save_llm_analysis(
    record: TranscriptRecord,
    analysis: Dict[str, Any],
) -> Path:
    """
    Save LLM analysis JSON to data/analysis/.
    """
    video_id = record.video_id or "unknown_id"
    ts = analysis.get("_meta", {}).get("analyzed_at") or datetime.now(timezone.utc).isoformat()
    ts_safe = ts.replace(":", "-")

    filename = f"{record.source}_{video_id}_llm_analysis_{ts_safe}.json"
    output_path = ANALYSIS_DIR / filename

    logger.info(f"Saving LLM analysis to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    logger.info("LLM analysis saved.")
    return output_path