from pathlib import Path
from src.youtube_downloader import download_audio
from src.transcriber import transcribe_file

url = "https://www.youtube.com/watch?v=9FuNtfsnRNo"

audio_path = download_audio(url)
print("Downloaded to: ", audio_path)

if audio_path:
    result = transcribe_file(Path(audio_path))
    print("Transcript text preview:")
    print(result.get("text", "")[:500])