"""
Test Gallery Scraper

Prototype to verify we can extract valid scene image URLs from IMDb's mediaindex page.
Target: https://www.imdb.com/title/{id}/mediaindex
"""

import requests
from bs4 import BeautifulSoup
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]


def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}


def get_scene_images(imdb_id: str, limit=5):
    url = f"https://www.imdb.com/title/{imdb_id}/mediaindex?ref_=tt_mv_close"
    print(f"Fetching: {url}")

    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # IMDb media gallery usually holds images in <img> tags inside 'media_index_imagelist'
        # or specific grid layouts.
        # Let's try finding images with specific attributes or classes.

        # Modern IMDb uses complex layout, often json embedded or specific classes.
        # Let's look for img tags with 'src' containing 'm.media-amazon.com/images/M/'
        # which are the actual image files.

        images = []
        for img in soup.find_all("img"):
            src = img.get("src", "")
            # Filter for likely content images (usually large thumbnails or full size)
            if (
                "media-amazon.com" in src and "UY" not in src
            ):  # 'UY' often denotes specific small icons stuff, but let's see results
                # Filter out tiny icons
                width = img.get("width")
                if width and int(width) < 50:
                    continue

                # IMDb thumbnails often have resizing params like "._V1_UY100_CR..._.jpg"
                # We can try to strip them to get full res: "._V1_.jpg"

                base_url = src.split("._V1")[0] + "._V1_.jpg"
                if base_url not in images:
                    images.append(base_url)

        return images[:limit]

    except Exception as e:
        print(f"Error: {e}")
        return []


# Test with Forrest Gump (tt0109830)
imgs = get_scene_images("tt0109830")
print(f"Found {len(imgs)} images:")
for i in imgs:
    print(i)
