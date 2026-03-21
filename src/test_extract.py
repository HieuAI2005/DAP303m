import sys
import logging

logging.basicConfig(level=logging.INFO)

# We can bypass indexing entirely since we just want to run the extraction method
from movierag.indexing.visual_indexer import VisualIndexer

# Create a dummy object and bind the method
indexer = object.__new__(VisualIndexer)

res = indexer.extract_clip_at_time(
    movie_id="tt0100405",
    start_time="02:47:30",
    end_time="02:55:00",
)
print(f"Extraction result: {res}")
