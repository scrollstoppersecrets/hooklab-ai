from pathlib import Path
from loguru import logger

from src.youtube_downloader import download_audio
from src.transcriber import transcribe_file
from src.storage import create_transcript_record, save_transcript_record

def main() -> None:
    url = "https://www.youtube.com/watch?v=9FuNtfsnRNo"

    logger.info(f"Downloadig audio for: {url}")
    audio_path = download_audio(url)
    if not audio_path:
        logger.error("Download failed, aborting.")
        return
    
    logger.info(f"Audio downloaded to : {audio_path}")
    logger.info(f"Starting trasncription...")

    result = transcribe_file(Path(audio_path))
    logger.info(f"Transcription fininshed")

    # Parse video_id from the file name 
    video_id = Path(audio_path).stem

    record = create_transcript_record(
        source="youtube",
        url=url,
        download_path=audio_path,
        transcript_result=result,
        video_id=video_id,
        title=None, # fill this later if you fetch the title
        metrics={} # filled later from your performance dataset
    )

    output_path = save_transcript_record(record)
    logger.info(f"Transcript JSON saved to: {output_path}")

if __name__ == "__main__":
    main()