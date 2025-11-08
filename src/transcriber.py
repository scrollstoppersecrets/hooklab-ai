from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import time 

import requests
from loguru import logger

from .config import get_assemblyai_api_key

# --- Assembly AI configuration ---
API_KEY = get_assemblyai_api_key()
API_BASE_URL = "https://api.assemblyai.com/v2"

# Headers for JSON-based endpoints
JSON_HEADERS = {
    "authorization": API_KEY,
    "content-type": "application/json",
}

# Headers for the upload endpoint
UPLOAD_HEADERS = {
    "authorization": API_KEY
}

# --- Internal helpers ----
def _read_file_in_chunks(file_path: Path, chunk_size: int = 5_242_880):
    """
    Generator that reads a file in chunks.

    AssemblyAI recommends uploading large files in chunks.
    Default chunk size: ~5 MB.
    """
    with file_path.open("rb") as f:
        while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data

def upload_file(file_path: Path) -> str:
    if not file_path.exists():
         raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Uploading file to AssemblyAI: {file_path}")
    upload_url = f"{API_BASE_URL}/upload"

    response = requests.post(
         upload_url,
         headers = UPLOAD_HEADERS,
         data = _read_file_in_chunks(file_path)
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error(
              f"Upload failed with status {response.status_code}: {response.text}"
        )
        raise exc
    
    data = response.json()
    audio_url = data.get("upload_url")

    if not audio_url:
         logger.error(f"Unexpected upload response format: {data}")
         raise RuntimeError("AssemblyAI upload failed: missing 'upload_url'")

    logger.info(f"Upload successful. audio_url: {audio_url}")
    return audio_url

def request_transcription(audio_url: str, **extra_options: Any) -> str:
    # Transcription request in AssemblyAI
    logger.info("Creating trascription request ...")
    endpoint = f"{API_BASE_URL}/transcript"

    payload: Dict[str, Any] = {
         "audio_url": audio_url
    }

    # Allow future extentions (sentiment, topics, etc.)
    payload.update(extra_options)

    response = requests.post(endpoint, json=payload, headers=JSON_HEADERS)

    try:
         response.raise_for_status()
    except requests.HTTPError as exc:
         logger.error(
              f"Transcription request failed with status {response.status_code}:"
              f"{response.text}"
         )
    data = response.json()
    transcript_id = data.get("id")

    if not transcript_id:
         logger.error(f"Unexpected transcription responce format: {data}")
         raise RuntimeError("AssemblyAI transcription request failed: missing 'id'")
    
    logger.info(f"Transcription request. transcript_id: {transcript_id}")
    return transcript_id

def wait_for_transcription(
    transcript_id: str,
    poll_interval: int = 3,
    timeout_seconds: int = 900, 
) -> Dict[str, Any]:
    """
    Poll AssemblyAI untill the transcription is complete or fails
    :param transcript_id: ID returned from request_transcription()
    :param poll_interval: Seconds between polls.
    :param timeout_seconds: Max time to wait before giving up.
    :return: The full transcription JSON response.
    """
    logger.info(
        f"Polling transction status for id={transcript_id} "
        f"(every {poll_interval}s, timeout {timeout_seconds}s)..."
    )

    endpoint = f"{API_BASE_URL}/transcript/{transcript_id}"
    start_time = time.time()

    while True:
        response = requests.get(endpoint, headers=JSON_HEADERS)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                f"Polling failed with status {response.status_code}: {response.text}"
            )
            raise exc

        data = response.json()
        status = data.get("status")

        if status == "completed":
             logger.info("Transcription completed successfully.")
             return data
        
        if status == "error":
            logger.error(
                f"Timeout waiting for transformation {transcript_id}."
                f"Last know status: {status}"
            )
            raise TimeoutError(
                f"Transcription {transcript_id} did not complete within { timeout_seconds} sseconds"
            )
             
        logger.debug(f"Status={status}, waiting {poll_interval}s before next check...")
        time.sleep(poll_interval)

# ---  high-level Orchastrator ---
def transcribe_file(
    file_path: Path,
    poll_interval: int = 3,
    timeout_seconds: int = 900,
    **extra_options: Any      
)->Dict[str, Any]: 
    """
    High-level helper: upload local file -> create transcription -> wait for result.

    :param file_path: Path to the local audio/video file.
    :param poll_interval: Seconds between polling requests.
    :param timeout_seconds: Max time to wait for the transcription to complete.
    :param extra_options: Extra AssemblyAI options, e.g. sentiment_analysis=True.
    :return: The full transcription JSON.
    The transcript text is typically under result['text'].
    """
    logger.info(f"Starting full transcription pipeline for: {file_path}")

    audio_url = upload_file(file_path)
    transcript_id = request_transcription(audio_url, **extra_options)
    result = wait_for_transcription(
         transcript_id,
         poll_interval=poll_interval,
         timeout_seconds=timeout_seconds
    )

    logger.info("Full transcription pipeline completed.")
    return result