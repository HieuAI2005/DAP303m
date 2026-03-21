import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
r = requests.get("https://www.imdb.com/title/tt0109830/mediaindex", headers=headers)
with open("imdb_dump.html", "w", encoding="utf-8") as f:
    f.write(r.text)
