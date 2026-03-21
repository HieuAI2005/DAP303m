#!/usr/bin/env python3
"""
download_trailers.py
====================
Download official movie trailers via yt-dlp using IMDb IDs.

Strategy:
  1. Load IMDb IDs from trailer_progress.json
  2. Fetch movie title via IMDb page (no API key needed)
  3. Search YouTube: "{title} official trailer" using yt-dlp
  4. Download to data/trailers/{imdb_id}.mp4
  5. Skip already downloaded

Usage:
    python scripts/download_trailers.py --check          # Show status
    python scripts/download_trailers.py --execute        # Download all
    python scripts/download_trailers.py --execute --limit 10  # Test 10
    python scripts/download_trailers.py --execute --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrailerDownloader")

PROGRESS_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "trailer_progress.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "trailers"
IMDB_TITLES_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "imdb_titles_cache.json"

# Known titles from our dataset
KNOWN_TITLES = {
    "tt0111161": "The Shawshank Redemption",
    "tt0071562": "The Godfather Part II",
    "tt0468569": "The Dark Knight",
    "tt0060196": "The Good the Bad and the Ugly",
    "tt0167260": "The Lord of the Rings The Return of the King",
    "tt0050083": "12 Angry Men",
    "tt0108052": "Schindler's List",
    "tt0137523": "Fight Club",
    "tt0120737": "The Lord of the Rings The Fellowship of the Ring",
    "tt0080684": "The Empire Strikes Back",
    "tt0167261": "The Lord of the Rings The Two Towers",
    "tt1375666": "Inception",
    "tt0133093": "The Matrix",
    "tt0047478": "Seven Samurai",
    "tt0114369": "Se7en",
    "tt0317248": "City of God",
    "tt0102926": "The Silence of the Lambs",
    "tt0099685": "Goodfellas",
    "tt0038650": "It's a Wonderful Life",
    "tt0120815": "Saving Private Ryan",
    "tt0076759": "Star Wars",
    "tt0245429": "Spirited Away",
    "tt0054215": "Psycho",
    "tt0118799": "Life is Beautiful",
    "tt0120689": "The Green Mile",
    "tt0816692": "Interstellar",
    "tt0056058": "Harakiri",
    "tt0114814": "The Usual Suspects",
    "tt0034583": "Casablanca",
    "tt0021749": "City Lights",
    "tt0120586": "American History X",
    "tt0082971": "Raiders of the Lost Ark",
    "tt0032553": "The Great Dictator",
    "tt0253474": "The Pianist",
    "tt0031381": "Gone with the Wind",
    "tt0043014": "Sunset Boulevard",
    "tt1675434": "The Intouchables",
    "tt0027977": "Modern Times",
    "tt0051201": "Witness for the Prosecution",
    "tt0064116": "Once Upon a Time in the West",
    "tt0407887": "The Departed",
    "tt0052357": "Vertigo",
    "tt0025316": "It Happened One Night",
    "tt0361748": "Inglourious Basterds",
    "tt0172495": "Gladiator",
    "tt2582802": "Whiplash",
    "tt0482571": "The Prestige",
    "tt1853728": "Django Unchained",
    "tt0405094": "The Lives of Others",
    "tt1345836": "The Dark Knight Rises",
    "tt0088763": "Back to the Future",
    "tt0986264": "Taare Zameen Par",
    "tt0062622": "2001 A Space Odyssey",
    "tt0209144": "Memento",
    "tt0071853": "Monty Python and the Holy Grail",
    "tt0119217": "Good Will Hunting",
    "tt2106476": "The Hunt",
    "tt0078748": "Alien",
    "tt0078788": "Apocalypse Now",
    "tt0092005": "Stand by Me",
    "tt1049413": "Up",
    "tt0095765": "Cinema Paradiso",
    "tt1392190": "Mad Max Fury Road",
    "tt2024544": "12 Years a Slave",
    "tt0095327": "Grave of the Fireflies",
    "tt0087843": "Once Upon a Time in America",
    "tt0086190": "Star Wars Return of the Jedi",
    "tt3011894": "Wild Tales",
    "tt0107290": "Jurassic Park",
    "tt0112573": "Braveheart",
    "tt0096283": "My Neighbor Totoro",
    "tt0119488": "L.A. Confidential",
    "tt0089881": "Ran",
    "tt1187043": "3 Idiots",
    "tt0457430": "Pan's Labyrinth",
    "tt0338013": "Eternal Sunshine of the Spotless Mind",
    "tt0364569": "Oldboy",
    "tt0036775": "Double Indemnity",
    "tt0090605": "Aliens",
    "tt1291584": "Warrior",
    "tt1205489": "Gone Girl",
    "tt1187120": "The Dark Knight",
    "tt1255953": "Incendies",
    "tt0119698": "Princess Mononoke",
    "tt0033467": "Citizen Kane",
    "tt1950186": "Ford v Ferrari",
    "tt4154796": "Avengers Endgame",
    "tt4154756": "Avengers Infinity War",
    "tt3783958": "La La Land",
    "tt0848228": "The Avengers",
    "tt2015381": "Guardians of the Galaxy",
    "tt0082096": "Das Boot",
    "tt2278388": "The Grand Budapest Hotel",
    "tt1832382": "A Separation",
    "tt6751668": "Parasite",
    "tt3170832": "Room",
    "tt2562232": "Birdman",
    "tt1291877": "Black Swan",
    "tt1673434": "The Social Network",
    "tt0405159": "Million Dollar Baby",
    "tt0371746": "Iron Man",
    "tt0386117": "Hot Fuzz",
    "tt0800369": "Thor",
    "tt0458339": "Captain America The First Avenger",
    "tt1211837": "Doctor Strange",
    "tt2395427": "Avengers Age of Ultron",
    "tt3498820": "Captain America Civil War",
    "tt1399103": "Thor The Dark World",
    "tt0458528": "Captain America The Winter Soldier",
    "tt1219289": "Limitless",
    "tt0440963": "The Bourne Identity",
    "tt0372183": "The Bourne Supremacy",
    "tt0479143": "The Bourne Ultimatum",
    "tt0258463": "The Bourne Identity",
    "tt0816711": "Sherlock Holmes",
    "tt1074638": "Skyfall",
    "tt0381061": "Casino Royale",
    "tt0758758": "Into the Wild",
    "tt1659337": "The Perks of Being a Wallflower",
    "tt0993846": "The Wolf of Wall Street",
    "tt1375670": "The Help",
    "tt1280558": "Rush",
    "tt0443453": "Hairspray",
    "tt0421715": "The Road",
    "tt0401792": "Sin City",
    "tt0457508": "Sweeney Todd The Demon Barber of Fleet Street",
    "tt0268978": "A Beautiful Mind",
    "tt0363163": "Downfall",
    "tt0353969": "Anchorman The Legend of Ron Burgundy",
    "tt0180093": "Requiem for a Dream",
    "tt0166924": "Mulholland Drive",
    "tt0379725": "Crash",
    "tt0266697": "Kill Bill Volume 1",
    "tt0208092": "Snatch",
    "tt0120735": "Lock Stock and Two Smoking Barrels",
    "tt0139654": "American Pie",
    "tt0120382": "The Truman Show",
    "tt0107048": "Groundhog Day",
    "tt0245712": "Amélie",
    "tt0082398": "Blade Runner",
    "tt0086879": "Scarface",
    "tt0112641": "Casino",
    "tt0060107": "Blow-Up",
    "tt0114388": "Heat",
    "tt0118849": "The Full Monty",
    "tt0095953": "Rain Man",
    "tt0103064": "Terminator 2 Judgment Day",
    "tt0096895": "Back to the Future Part II",
    "tt0088247": "The Terminator",
    "tt0166786": "There Will Be Blood",
    "tt0268380": "Chicago",
    "tt2084970": "The Imitation Game",
    "tt0275847": "About Schmidt",
    "tt2980516": "Still Alice",
    "tt1748122": "Boyhood",
    "tt4154664": "Captain Marvel",
    "tt1843866": "Captain America The Winter Soldier",
    "tt2229499": "Her",
    "tt1825683": "Black Panther",
    "tt9362722": "Spider-Man Across the Spider-Verse",
    "tt10872600": "Spider-Man No Way Home",
    "tt6320628": "Spider-Man Far From Home",
    "tt6966692": "Green Book",
    "tt1877830": "The Batman",
    "tt7286456": "Joker",
    "tt1160419": "Dune",
    "tt5289954": "Doctor Strange in the Multiverse of Madness",
    "tt6264654": "Thor Ragnarok",
    "tt2527336": "Star Wars The Last Jedi",
    "tt2488496": "Star Wars The Force Awakens",
    "tt2527338": "Star Wars The Rise of Skywalker",
    "tt9114286": "Black Widow",
    "tt4729430": "Doctor Strange",
    "tt9032400": "Eternals",
    "tt10648342": "Thor Love and Thunder",
    "tt5013056": "Dunkirk",
    "tt7131622": "Once Upon a Time in Hollywood",
    "tt1872194": "The Power of the Dog",
    "tt0114709": "Toy Story",
    "tt1745960": "Top Gun Maverick",
    "tt3107288": "The Big Short",
    "tt1663202": "The Revenant",
    "tt2345759": "Coco",
    "tt2724064": "Frozen",
    "tt0373889": "Batman Begins",
    "tt0304141": "The Incredibles",
    "tt3501632": "Thor",
    "tt1201607": "Harry Potter and the Deathly Hallows Part 2",
    "tt0926084": "Fantastic Mr Fox",
    "tt2119532": "Hacksaw Ridge",
    "tt0478970": "Ant-Man",
}


def get_title(imdb_id: str, cache: dict) -> str:
    """Get movie title from cache or known titles."""
    if imdb_id in cache:
        return cache[imdb_id]
    if imdb_id in KNOWN_TITLES:
        return KNOWN_TITLES[imdb_id]
    return imdb_id  # fallback to IMDb ID


def download_trailer(imdb_id: str, title: str, output_dir: Path) -> tuple[str, bool]:
    """Download trailer using yt-dlp YouTube search."""
    out_path = output_dir / f"{imdb_id}.mp4"
    if out_path.exists() and out_path.stat().st_size > 100_000:
        return imdb_id, True

    # Build YouTube search query
    search_query = f"{title} official trailer"
    search_url = f"ytsearch1:{search_query}"

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "best[height<=480]/best",
        "-o", str(out_path.resolve()),
        "--no-check-certificates",
        "--user-agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "--max-filesize", "200m",   # Skip anything > 200MB
        search_url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Wait briefly for file to finalize
        time.sleep(0.5)
        return imdb_id, out_path.exists() and out_path.stat().st_size > 100_000
    except subprocess.TimeoutExpired:
        if out_path.exists():
            out_path.unlink()
        return imdb_id, False
    except Exception:
        return imdb_id, False


def check_status():
    """Show current download status."""
    progress = json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}
    all_ids = list(set(progress.get("downloaded", []) + progress.get("indexed", [])))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = {f.stem for f in OUTPUT_DIR.glob("*.mp4") if f.stat().st_size > 100_000}

    logger.info(f"Trailer download status:")
    logger.info(f"  Total IMDb IDs catalogued: {len(all_ids)}")
    logger.info(f"  Already downloaded: {len(downloaded)}")
    logger.info(f"  Remaining: {len(all_ids) - len(downloaded)}")
    logger.info(f"  Output dir: {OUTPUT_DIR}")


def download_all(limit: int = 0, workers: int = 3):
    """Download all trailers using parallel workers."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress = json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}
    all_ids = list(set(progress.get("downloaded", []) + progress.get("indexed", [])))

    # Load title cache
    cache = {}
    if IMDB_TITLES_FILE.exists():
        cache = json.loads(IMDB_TITLES_FILE.read_text())

    downloaded = {f.stem for f in OUTPUT_DIR.glob("*.mp4") if f.stat().st_size > 100_000}
    remaining = [mid for mid in all_ids if mid not in downloaded]

    logger.info(f"Total: {len(all_ids)} | Downloaded: {len(downloaded)} | Remaining: {len(remaining)}")

    if not remaining:
        logger.info("All trailers already downloaded!")
        return

    if limit:
        remaining = remaining[:limit]
    logger.info(f"Will download {len(remaining)} trailers ({workers} workers)...")

    success, fail = 0, 0
    total = len(remaining)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_trailer, mid, get_title(mid, cache), OUTPUT_DIR): mid
            for mid in remaining
        }

        for i, future in enumerate(as_completed(futures)):
            mid, ok = future.result()
            if ok:
                success += 1
                logger.info(f"  ✅ {mid}: {get_title(mid, cache)}")
            else:
                fail += 1
                logger.warning(f"  ❌ {mid}: failed")

            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"  Progress: {i+1}/{total} | OK:{success} FAIL:{fail}")

    logger.info(f"Done: {success}/{total} downloaded, {fail} failed")


def main():
    parser = argparse.ArgumentParser(description="Download movie trailers via yt-dlp")
    parser.add_argument("--execute", action="store_true", help="Actually download")
    parser.add_argument("--limit", type=int, default=0, help="Limit (0=all)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    args = parser.parse_args()

    if args.execute:
        download_all(limit=args.limit, workers=args.workers)
    else:
        check_status()


if __name__ == "__main__":
    main()
