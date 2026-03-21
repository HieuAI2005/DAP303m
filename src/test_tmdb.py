import sys, logging, os
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO)
import env_loader
from preprocess_data.extractors.metadata_extractor import MetadataCrawler
from preprocess_data.config import PreprocessConfig as Cfg

crawler = MetadataCrawler()
Cfg.META_SEARCH_DIRS = []  # Force web crawl

print("Testing TMDB API on Inception (tt1375666)...")
meta = crawler.crawl("tt1375666", force=True)

print("\n--- RESULTS ---")
print("Title:", meta.get("title"))
print("Genres:", meta.get("genres"))
print("Cast Count:", len(meta.get("cast", [])))

if len(meta.get("cast", [])) > 0:
    first_actor = meta["cast"][0]
    print(f"First Actor: {first_actor['name']} (ID: {first_actor['id']})")
    print("Image Path:", first_actor.get("profile_image"))

target = Cfg.get_actor_references_dir() / "tt1375666"
print(f"Actor Images Downloaded: {len(os.listdir(target)) if target.exists() else 0}")
