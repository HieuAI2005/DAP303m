"""
Build Visual Index Script

This script builds the FAISS visual index from MovieNet keyframes.
Run this once to create the index, then use search_demo.py for queries.

Usage:
    python -m movierag.scripts.build_index --data-dir data/movienet_tools/tests/data --sample
    python -m movierag.scripts.build_index --data-dir data/movienet --full
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from movierag.data.movienet_loader import MovieNetLoader
from movierag.indexing.visual_indexer import VisualIndexer
from movierag.config import get_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_index(data_dir: str, index_dir: str, use_sample: bool = False) -> None:
    """
    Build the visual index.

    Args:
        data_dir: Path to MovieNet data directory
        index_dir: Path to save the index
        use_sample: Whether to use sample data mode
    """
    logger.info("=" * 50)
    logger.info("MovieRAG Visual Index Builder")
    logger.info("=" * 50)

    # Initialize loader
    logger.info(f"Loading data from: {data_dir}")
    loader = MovieNetLoader(data_dir, use_sample=use_sample)

    # Get statistics
    stats = loader.get_statistics()
    logger.info(f"Dataset stats: {stats}")

    # Collect all keyframes
    logger.info("Collecting keyframes...")
    items = list(loader.get_all_keyframes())

    if not items:
        logger.error("No keyframes found! Check your data directory.")
        return

    logger.info(f"Found {len(items)} keyframes to index")

    # Initialize indexer
    indexer = VisualIndexer(index_dir=index_dir, index_name="visual_index")

    # Build index
    logger.info("Building index (this may take a while)...")
    indexer.build_index(items)

    # Print stats
    index_stats = indexer.get_statistics()
    logger.info(f"Index stats: {index_stats}")

    logger.info("=" * 50)
    logger.info("Index built successfully!")
    logger.info(f"Index saved to: {index_dir}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Build MovieRAG visual index from MovieNet keyframes"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to MovieNet data directory"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/indexes",
        help="Path to save the index (default: data/indexes)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data mode (for movienet-tools test data)",
    )

    args = parser.parse_args()

    build_index(
        data_dir=args.data_dir, index_dir=args.index_dir, use_sample=args.sample
    )


if __name__ == "__main__":
    main()
