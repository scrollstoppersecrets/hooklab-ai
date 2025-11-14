from pathlib import Path
import os

from dotenv import load_dotenv


# --- Paths & Base Directories ----

# Resolve the project root (folder containgn 'src/' and 'data/')
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the .env file in the project root
ENV_PATH = BASE_DIR / ".env"

# ---- Environment Variable ----

# Load variable from .evn if it exits
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # Print warning not fatal
    print(f"[config] WARNING: .env file not found ar {ENV_PATH}")

def get_assemblyai_api_key() -> str:

    # Read AssemblyAI API key from enviroment 
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        #Fail fast, don't run without valid api key
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set."
            "Create a .env file in the project root with ASSEMBLYAI_API_KEY=you_api_key"
        )
    return api_key

def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )
    return key

# --- Data Directories ----
DATA_DIR = BASE_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
ANALYSIS_DIR = DATA_DIR / "analysis"
PROMPTS_DIR = DATA_DIR / "prompts"
METRICS_DIR = DATA_DIR / "metrics"
LOGS_DIR = BASE_DIR / "logs" 

# Make sure dir's exits
for directory in (
    DATA_DIR, 
    DOWNLOADS_DIR, 
    TRANSCRIPTS_DIR, 
    ANALYSIS_DIR, 
    METRICS_DIR,
    LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# --- YouTube / yt-dpl Defaults --- 

# Template for how yt-dlp should na,e downloaded files.
# Example results: data/downloads/ewrTU6755lkjUYuuY876.m4a
YTDLP_OUTPUT_TEMPLATE = str(DOWNLOADS_DIR / "%(id)s.%(ext)s")

# Optional cookies file for YouTube (Option B - cookies.txt)
# Set COOKIEFILE in .env, e.g. COOKIEFILE=./cookies.txt
COOKIEFILE = os.getenv("COOKIEFILE")

DEFAULT_AUDIO_FORMAT = "bestaudio/best"
DEFAULT_METRICS_CSV = METRICS_DIR / "n8nYouTubeCombinedTop5Bottom5.csv"

# ----- Batch / throttling settings ----
# Random sleep between video downloads (in seconds)
MIN_DOWNLOAD_SLEEP_SECONDS = float(os.getenv("MIN_DOWNLOAD_SLEEP_SECONDS", "2"))
MAX_DOWNLOAD_SLEEP_SECONDS = float(os.getenv("MAX_DOWNLOAD_SLEEP_SECONDS", "8"))

MAX_VIDEOS_PER_RUN = int(os.getenv("MAX_VIDEOS_PER_RUN", "10")) # 0 = no limit