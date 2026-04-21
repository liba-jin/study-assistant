"""YouTube Data API v3 wrapper for finding learning resources."""

import requests
from config import YOUTUBE_API_KEY

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def search_courses(query: str, max_results: int = 5) -> list[dict]:
    """
    Search YouTube for educational videos/courses on a topic.
    Returns list of {title, url, channel, description}.
    """
    if not YOUTUBE_API_KEY:
        return [{"error": "YouTube API key not configured. Add YOUTUBE_API_KEY to .env"}]

    params = {
        "part": "snippet",
        "q": f"{query} tutorial course",
        "type": "video",
        "maxResults": max_results,
        "order": "relevance",
        "videoDuration": "long",  # Prefer longer educational content
        "key": YOUTUBE_API_KEY,
    }

    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return [{"error": f"YouTube API request failed: {str(e)}"}]

    results = []
    for item in data.get("items", []):
        snippet = item["snippet"]
        video_id = item["id"]["videoId"]
        results.append({
            "title": snippet["title"],
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": snippet["channelTitle"],
            "description": snippet["description"][:200],
        })

    return results
