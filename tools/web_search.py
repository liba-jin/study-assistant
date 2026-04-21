"""Brave Search API wrapper for finding learning resources and counterexamples."""

import requests
from config import BRAVE_API_KEY

SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for articles, MOOCs, and learning resources.
    Returns list of {title, url, description}.
    """
    if not BRAVE_API_KEY:
        return [{"error": "Brave Search API key not configured. Add BRAVE_API_KEY to .env"}]

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": max_results,
    }

    try:
        resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return [{"error": f"Brave Search request failed: {str(e)}"}]

    results = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", "")[:300],
        })

    return results


def search_counterexample(claim: str, topic: str) -> list[dict]:
    """
    Search for counterexamples or contradicting evidence for a claim.
    Used by the Test/Teacher agent to challenge user understanding.
    """
    query = f"{topic} counterexample OR exception OR limitation OR criticism \"{claim}\""
    return search_web(query, max_results=3)
