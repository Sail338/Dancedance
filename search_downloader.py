from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pytube import YouTube

api_file = open("API_KEY.txt", "r")
API_KEY = api_file.readline().strip()

def youtube_search(query, maxResults):
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=maxResults
    ).execute()

    videos = []

    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            print("ayy a video")
            videos.append(search_result['id']['videoId'])
    print(videos)
    return videos

def download_videos(videos, path_prefix=''):
    for vid in videos:
        vid_name = "http://youtube.com/watch?v=" + vid
        yt = YouTube(vid_name)
        print(yt.title)
        streams = yt.streams.all()

        #this is bad but the order_by in't work
        for stream in streams:
            if stream.resolution == '480p':
                stream.download(path_prefix)
                return

        for stream in streams:
            if stream.resolution == '720p':
                stream.download(path_prefix)
                return

        print("no acceptible resolution for video found")


def search_n_dl(query, maxResults, path_prefix=''):
    videos = youtube_search(query, maxResults)
    download_videos(videos, path_prefix)

if __name__ == "__main__":
    videos = youtube_search("hello", 3)
    download_videos(videos, 'test')
