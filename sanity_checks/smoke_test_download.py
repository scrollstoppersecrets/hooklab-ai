from src.youtube_downloader import download_audio

path = download_audio("https://www.youtube.com/watch?v=9FuNtfsnRNo")
if path is None:
    print("Download failed")
else:
    print("Downloaded to: ", path)