"""
Metadata Crawler — IMDB/TMDB → meta JSON

Wraps movienet_tools crawler to fetch movie metadata from online sources
and save as meta JSON compatible with the existing pipeline.

For new videos when IMDB ID is known.
"""

import logging
import json
import re
from typing import Dict, Any, List

import requests
from bs4 import BeautifulSoup

from preprocess_data.config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class MetadataCrawler:
    """Fetch movie metadata from IMDB/TMDB and save as JSON."""

    def __init__(self):
        self._imdb_crawler = None
        self._tmdb_crawler = None
        self._init_crawlers()

    def _init_crawlers(self):
        """Initialize movienet crawlers if available."""
        import sys
        try:
            crawler_path = Cfg.GLOBAL_DATA_DIR / "movienet_tools"
            if crawler_path.exists():
                sys.path.insert(0, str(crawler_path))
                from movienet.tools.crawler.imdb_crawler import IMDBCrawler

                self._imdb_crawler = IMDBCrawler()
                logger.debug("  IMDB crawler initialized")

        except Exception as e:
            logger.debug(f"  IMDB crawler not available: {e}")

        try:
            from movienet.tools.crawler.tmdb_crawler import TMDBCrawler

            self._tmdb_crawler = TMDBCrawler()
            logger.debug("  TMDB crawler initialized")
        except Exception:
            pass

    def crawl(self, movie_id: str, force: bool = False) -> Dict:
        """
        Crawl metadata for a movie and save as JSON.

        Args:
            movie_id: IMDB ID (e.g., 'tt0120338') or custom ID
            force: Overwrite existing meta file

        Returns: metadata dict
        """
        # Check local output dir
        Cfg.get_meta_dir().mkdir(parents=True, exist_ok=True)
        meta_path = Cfg.get_meta_dir() / f"{movie_id}.json"

        if meta_path.exists() and not force:
            logger.info(f"  ⏩ Meta already exists in output: {meta_path.name}")
            data = self._finalize_meta(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
            self._extract_script(movie_id, data.get("title", movie_id))
            return data

        # Check global search dirs
        for search_dir in Cfg.META_SEARCH_DIRS:
            candidate = search_dir / f"{movie_id}.json"
            if candidate.exists():
                logger.info(f"  📥 Found existing meta in {search_dir.name}: {candidate.name}")
                data = self._finalize_meta(
                    json.loads(candidate.read_text(encoding="utf-8"))
                )
                # Copy to local output dir
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self._extract_script(movie_id, data.get("title", movie_id))
                return data

        logger.info(f"  🌐 Crawling metadata: {movie_id}")


        meta = {"imdb_id": movie_id}
        fetch_success = False

        if movie_id.startswith("tt"):
            # 1. Try TMDB API first (richer data + actor profile images)
            fetch_success = self._try_tmdb(movie_id, meta)
            
            # 2. Fallback to OMDB if TMDB fails
            if not fetch_success:
                fetch_success = self._try_omdb(movie_id, meta)

        # 3. Fallback to dummy data
        if not fetch_success or "title" not in meta:
            logger.info(f"  Creating minimal meta stub")
            meta = {
                "imdb_id": movie_id,
                "title": movie_id.replace("_", " ").title(),
                "genres": [],
                "cast": [],
                "auto_generated": True,
            }

        # 4. Enhance with IMDB scraping when plot/cast/director are weak.
        if self._imdb_crawler and self._needs_imdb_backfill(meta):
            logger.info("  Enhancing metadata with direct IMDB scrape...")
            try:
                scraped = self._crawl_imdb(movie_id)
                meta = self._merge_meta_from_imdb(meta, scraped)
            except Exception as e:
                logger.debug(f"  IMDB scrape enhancement failed: {e}")

        meta = self._finalize_meta(meta)

        # Save Meta
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"  ✅ Meta saved: {meta.get('title', movie_id)} → {meta_path}")
        
        # 5. Attempt to automatically scrape the Script if it doesn't exist!
        self._extract_script(movie_id, meta.get("title", movie_id))
        
        return meta

    @staticmethod
    def _clean_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @classmethod
    def _normalize_person_name(cls, value: Any) -> str:
        text = cls._clean_text(value)
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _normalize_character_name(cls, value: Any) -> str:
        text = cls._clean_text(value)
        if not text or text.upper() == "N/A":
            return "Unknown"
        text = re.sub(r"\s+", " ", text)
        return text.strip(" -") or "Unknown"

    @classmethod
    def _normalize_cast_entry(cls, raw: Dict[str, Any]) -> Dict[str, str] | None:
        if not isinstance(raw, dict):
            return None
        name = cls._normalize_person_name(raw.get("name") or raw.get("actor"))
        if not name:
            return None
        actor_id = cls._clean_text(raw.get("id")) or name.lower().replace(" ", "_")
        character = cls._normalize_character_name(raw.get("character"))
        entry = {
            "id": actor_id,
            "name": name,
            "character": character,
        }
        profile_image = cls._clean_text(raw.get("profile_image"))
        if profile_image:
            entry["profile_image"] = profile_image
        return entry

    @classmethod
    def _normalize_cast(cls, cast_entries: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        seen = set()
        for raw in cast_entries or []:
            entry = cls._normalize_cast_entry(raw)
            if not entry:
                continue
            key = cls._clean_text(entry["id"]).lower() or cls._clean_text(entry["name"]).lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(entry)
        return normalized

    @classmethod
    def _normalize_director(cls, director_value: Any) -> List[Dict[str, str]]:
        if not director_value:
            return []
        values = director_value if isinstance(director_value, list) else [director_value]
        normalized: List[Dict[str, str]] = []
        seen = set()
        for raw in values:
            if isinstance(raw, dict):
                name = cls._normalize_person_name(raw.get("name"))
                director_id = cls._clean_text(raw.get("id")) or name.lower().replace(" ", "_")
            else:
                name = cls._normalize_person_name(raw)
                director_id = name.lower().replace(" ", "_")
            if not name:
                continue
            key = director_id.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"id": director_id, "name": name})
        return normalized

    @classmethod
    def _is_cast_weak(cls, meta: Dict[str, Any]) -> bool:
        cast = cls._normalize_cast(meta.get("cast"))
        if not cast:
            return True
        known_characters = sum(
            1 for cast_entry in cast if cls._normalize_character_name(cast_entry.get("character")) != "Unknown"
        )
        return known_characters < max(1, min(3, len(cast)))

    @classmethod
    def _needs_imdb_backfill(cls, meta: Dict[str, Any]) -> bool:
        if not meta.get("storyline") or not meta.get("synopsis"):
            return True
        if not cls._normalize_director(meta.get("director")):
            return True
        if cls._is_cast_weak(meta):
            return True
        return False

    @classmethod
    def _merge_cast(cls, existing_cast: Any, scraped_cast: Any) -> List[Dict[str, str]]:
        existing = cls._normalize_cast(existing_cast)
        scraped = cls._normalize_cast(scraped_cast)
        by_name = {cls._clean_text(item["name"]).lower(): dict(item) for item in existing}
        order = [cls._clean_text(item["name"]).lower() for item in existing]

        for item in scraped:
            key = cls._clean_text(item["name"]).lower()
            current = by_name.get(key)
            if current is None:
                by_name[key] = dict(item)
                order.append(key)
                continue
            current_char = cls._normalize_character_name(current.get("character"))
            new_char = cls._normalize_character_name(item.get("character"))
            if current_char == "Unknown" and new_char != "Unknown":
                current["character"] = new_char
            if not current.get("id") and item.get("id"):
                current["id"] = item["id"]
            if not current.get("profile_image") and item.get("profile_image"):
                current["profile_image"] = item["profile_image"]

        return [by_name[key] for key in order if key in by_name]

    @classmethod
    def _merge_meta_from_imdb(cls, meta: Dict[str, Any], scraped: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(meta)
        for key in ("title", "country", "storyline", "synopsis"):
            if not merged.get(key) and scraped.get(key):
                merged[key] = scraped.get(key)
        if not merged.get("genres") and scraped.get("genres"):
            merged["genres"] = scraped.get("genres", [])
        if not cls._normalize_director(merged.get("director")) and scraped.get("director"):
            merged["director"] = scraped.get("director")
        if cls._is_cast_weak(merged) and scraped.get("cast"):
            merged["cast"] = cls._merge_cast(merged.get("cast"), scraped.get("cast"))
        elif scraped.get("cast"):
            merged["cast"] = cls._merge_cast(merged.get("cast"), scraped.get("cast"))
        return merged

    @classmethod
    def _finalize_meta(cls, meta: Dict[str, Any]) -> Dict[str, Any]:
        finalized = dict(meta or {})
        finalized["title"] = cls._clean_text(finalized.get("title")) or finalized.get("imdb_id", "")
        finalized["genres"] = [
            cls._clean_text(genre)
            for genre in finalized.get("genres", []) or []
            if cls._clean_text(genre)
        ]
        finalized["cast"] = cls._normalize_cast(finalized.get("cast"))
        finalized["director"] = cls._normalize_director(finalized.get("director"))
        if "storyline" in finalized:
            finalized["storyline"] = cls._clean_text(finalized.get("storyline"))
        if "synopsis" in finalized:
            finalized["synopsis"] = cls._clean_text(finalized.get("synopsis"))
        return finalized

    def _try_tmdb(self, imdb_id: str, meta: Dict) -> bool:
        """Try fetching from TMDB and download actor images."""
        import os
        import requests
        
        tmdb_key = os.environ.get("TMDB_API_KEY")
        if not tmdb_key:
            logger.info("  TMDB_API_KEY not found. Skipping TMDB fetch.")
            return False
            
        logger.info("  Calling TMDB API...")
        try:
            # Step A: Find TMDB ID from IMDB ID
            find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={tmdb_key}&external_source=imdb_id"
            resp = requests.get(find_url, timeout=10)
            if resp.status_code != 200: return False
            
            data = resp.json()
            movies = data.get("movie_results", [])
            if not movies: return False
            
            tmdb_id = movies[0]["id"]
            meta["title"] = movies[0].get("title", imdb_id)
            meta["storyline"] = movies[0].get("overview", "")
            
            # Step B: Fetch detailed credits & genres
            details_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={tmdb_key}&append_to_response=credits"
            resp_det = requests.get(details_url, timeout=10)
            if resp_det.status_code != 200: return True # At least we got title
            
            det_data = resp_det.json()
            meta["genres"] = [g["name"] for g in det_data.get("genres", [])]
            
            credits = det_data.get("credits", {})
            crew = credits.get("crew", [])
            directors = [c["name"] for c in crew if c.get("job") == "Director"]
            if directors:
                meta["director"] = [{"name": directors[0]}]
                
            cast_data = credits.get("cast", [])
            cast = []
            
            # Setup image download directory
            img_dir = Cfg.get_actor_references_dir() / imdb_id
            img_dir.mkdir(parents=True, exist_ok=True)
            
            # Limit to top 15 actors to save time/bandwidth
            for c in cast_data[:15]:
                actor_id = c["name"].lower().replace(" ", "_")
                actor_name = c["name"]
                character = c.get("character", "Unknown")
                profile_path = c.get("profile_path")
                
                local_img_path = None
                if profile_path:
                    # Download the image
                    img_url = f"https://image.tmdb.org/t/p/w185{profile_path}"
                    local_img = img_dir / f"{actor_id}.jpg"
                    if not local_img.exists():
                        try:
                            img_data = requests.get(img_url, timeout=5).content
                            local_img.write_bytes(img_data)
                        except Exception as e:
                            logger.debug(f"Failed to dl image for {actor_name}: {e}")
                    
                    if local_img.exists():
                        local_img_path = str(local_img)
                
                cast.append({
                    "id": actor_id,
                    "name": actor_name,
                    "character": character,
                    "profile_image": local_img_path
                })
                
            meta["cast"] = cast
            meta["auto_generated"] = False
            logger.info(f"  ✅ TMDB fetch success: {meta['title']} ({len(meta['cast'])} cast, downloaded images)")
            return True
        except Exception as e:
            logger.warning(f"  TMDB fetch failed: {e}")
            return False

    def _try_omdb(self, imdb_id: str, meta: Dict) -> bool:
        """Fallback to OMDB API."""
        import os
        import requests
        
        logger.info("  Calling OMDB API...")
        try:
            api_key = os.environ.get("OMDB_API_KEY", "trilogy")
            resp = requests.get(f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("Response") == "True":
                    meta["title"] = data.get("Title", imdb_id)
                    meta["genres"] = [g.strip() for g in data.get("Genre", "").split(",") if g.strip()]
                    
                    actors_str = data.get("Actors", "")
                    cast = []
                    for actor in actors_str.split(","):
                        actor = actor.strip()
                        if actor and actor != "N/A":
                            cast.append({
                                "id": actor.lower().replace(" ", "_"),
                                "name": actor,
                                "character": "Unknown"
                            })
                    meta["cast"] = cast
                    meta["director"] = [{"name": data.get("Director", "Unknown")}]
                    meta["storyline"] = data.get("Plot", "")
                    meta["auto_generated"] = False
                    logger.info(f"  ✅ OMDB fetch success: {meta['title']} ({len(meta['cast'])} cast)")
                    return True
            return False
        except Exception as e:
            logger.warning(f"  OMDB fetch failed: {e}")
            return False

    def _crawl_imdb(self, imdb_id: str) -> Dict:

        """Crawl all available data from IMDB."""
        meta = {"imdb_id": imdb_id, "auto_generated": True}

        # Home page: title, genres, storyline, country
        try:
            home = self._imdb_crawler.parse_home_page(imdb_id)
            meta.update(
                {
                    "title": home.get("title", imdb_id),
                    "genres": home.get("genres", []),
                    "country": home.get("country"),
                    "storyline": home.get("storyline"),
                }
            )
            logger.info(f"    Title: {meta.get('title')}")
        except Exception as e:
            logger.warning(f"    IMDB home page failed: {e}")

        # Credits page: director, cast
        try:
            credits = self._imdb_crawler.parse_credits_page(imdb_id)
            meta["director"] = credits.get("director")
            meta["cast"] = credits.get("cast", [])
            logger.info(f"    Cast: {len(meta.get('cast', []))} actors")
        except Exception as e:
            logger.warning(f"    IMDB credits page failed: {e}")

        # Synopsis
        try:
            synopsis = self._imdb_crawler.parse_synopsis(imdb_id)
            if synopsis.get("synopsis"):
                meta["synopsis"] = synopsis["synopsis"]
        except Exception as e:
            logger.debug(f"    IMDB synopsis failed: {e}")

        return meta

    def _extract_script(self, movie_id: str, title: str):
        """Attempts to scrape the movie script from IMSDb if it doesn't exist locally."""
        script_dir = Cfg.get_script_dir()
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{movie_id}.script"
        
        if script_path.exists():
            return
            
        logger.info(f"  📜 Attempting to crawl script for '{title}'...")
        try:
            # Format title for IMSDb (e.g. "The Godfather" -> "Godfather,-The")
            search_title = title.replace(" ", "-")
            if search_title.lower().startswith("the-"):
                search_title = search_title[4:] + ",-The"
            elif search_title.lower().startswith("a-"):
                search_title = search_title[2:] + ",-A"
                
            # Attempt IMSDb
            url = f"https://imsdb.com/scripts/{search_title}.html"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'lxml')
                # Scripts are typically in a <pre> tag inside a td.scrtext
                scrtext = soup.find('td', {'class': 'scrtext'})
                if scrtext:
                    pre = scrtext.find('pre')
                    if pre:
                        text = pre.get_text()
                        script_path.write_text(text, encoding="utf-8")
                        logger.info(f"  ✅ Script successfully downloaded to {script_path.name}")
                        return
                        
            # If IMSDb fails, we would try others (Daily Script, etc.). 
            # For brevity, we log success/failure of IMSDb.
            logger.info("  ⚠️ Could not find script online. Global reasoning will rely on Synopsis.")
            
        except Exception as e:
            logger.debug(f"  Script crawl failed: {e}")
