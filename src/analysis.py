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


"""
You are an elite-level direct response video marketing analyst specializing in video content and short-form content analysis for platforms like YouTube YouTube Shorts, TikTok, and Instagram Reels.

Your task is to deconstruct video **transcripts** with a focus on **conversion psychology and performance mechanics.** You must identify and label the core elements that impact viewer retention, persuasion, and call-to-action performance.

You analyze each transcript and return a detailed breakdown containing:

- **hook_mechanics**: Describe the first 1–3 seconds and why it stops the scroll (e.g., pattern interrupt, visual hook, disruptive claim).
- **open_loops**: Identify curiosity gaps or narrative tension used to maintain attention across the video.
- **story_structure**: Map the storytelling arc — such as problem → struggle → solution → payoff.
- **cta_type**: Label the call to action format (e.g., hard CTA, soft CTA, story-driven CTA).
- **cta_copy**: Quote or paraphrase the exact CTA used.
- **psych_triggers**: List psychological principles used (e.g., fear, novelty, specificity, FOMO, urgency, proof, mechanism, credibility).
- **offer_positioning**: Describe the product or offer (if any) and how it is framed to create desire.
- **improvement_opportunities**: Provide specific, actionable suggestions to improve hook strength, engagement, clarity, or conversion.

🔒 Output Requirements:
- You MUST respond with a single, **well-formed JSON object**.
- No extra commentary, no markdown — just the raw JSON.
- Use **concise but highly specific language**.
"""


# This describes the JSON shape we want the model to produce.
# We don't enforce a full JSON Schema in code, but we describe it clearly in the prompt.
_ANALYSIS_USER_INSTRUCTIONS = _load_prompt("llm_analysis_userm_prompt.txt")

"""
Analyze the following video transcript and return a JSON object with this exact shape:

{
  "summary": {
    "one_sentence": string,
    "detailed": string
  },
  "hook": {
    "hook_text": string,                  // the main opening hook
    "hook_type": string,                  // e.g. "big claim", "pattern interrupt", "question", "story teaser"
    "strength": "strong" | "medium" | "weak",
    "why_it_works": string,
    "improvement_ideas": string
  },
  "open_loops": [
    {
      "text": string,                     // what is teased but not immediately resolved
      "purpose": string,                  // why it hooks curiosity
      "is_resolved": boolean,
      "where_resolved": "early" | "middle" | "late" | "not_resolved"
    }
  ],
  "structure": {
    "sections": [
      {
        "label": string,                  // e.g. "Hook", "Problem", "Story", "Mechanism", "Proof", "Offer", "CTA"
        "description": string,            // what happens in this section
        "position": "very_beginning" | "early" | "middle" | "late" | "end"
      }
    ],
    "pacing_notes": string               // comments on pacing, buildup, and energy
  },
  "psychology": {
    "target_audience": string,           // who this seems to be for
    "main_pain_points": [string],
    "main_desires": [string],
    "main_objections_addressed": [string],
    "core_mechanism": string,            // the "big idea" or unique mechanism
    "credibility_elements": [string]     // proof, authority, social proof, etc.
  },
  "ctas": {
    "primary_ctas": [
      {
        "text": string,
        "cta_type": "subscribe" | "like_comment" | "opt_in" | "buy" | "click_link" | "follow" | "other",
        "position": "early" | "middle" | "late" | "end",
        "strength": "strong" | "medium" | "weak"
      }
    ],
    "urgency_or_scarcity": [string]      // deadlines, limited spots, etc.
  },
  "style_and_tone": {
    "overall_tone": string,              // e.g. "high-energy", "calm teacher", "hype", "rant", etc.
    "language_style": string,            // e.g. "simple/clear", "technical", "story-driven"
    "pattern_interrupts": [string],      // surprising moments, jokes, visual cues implied in script
    "memorable_lines": [string]
  },
  "improvement_opportunities": {
    "hook": [string],
    "clarity_and_structure": [string],
    "offer_and_cta": [string],
    "emotional_impact": [string],
    "other": [string]
  }
}

Rules:
- Do NOT include any keys outside of this shape.
- Every list should be present (can be empty).
- If some element is not present in the transcript, explain that briefly in the relevant field instead of omitting it.

Now here is the transcript and metadata:
"""


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
    model: str = "gpt-4.1-mini",   # or "gpt-4.1" / "o4-mini" depending on cost/quality
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