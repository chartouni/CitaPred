"""
Quick script to check how many AUB papers exist on Semantic Scholar.

This checks the 'total' count from the API without fetching all papers.
"""

import sys
from pathlib import Path
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.utils.logger import setup_logger

logger = setup_logger()


def get_paper_count(query: str, api_key: str) -> int:
    """
    Get total count of papers matching a query.

    Args:
        query: Search query
        api_key: Semantic Scholar API key

    Returns:
        Total count of papers
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    headers = {}
    if api_key:
        headers['x-api-key'] = api_key

    params = {
        "query": query,
        "limit": 1,  # Just need the count, not the actual papers
        "fields": "paperId"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        total = data.get('total', 0)

        return total

    except Exception as e:
        logger.error(f"Error: {e}")
        return 0


def main():
    """Check AUB paper counts for different queries."""

    logger.info("="*60)
    logger.info("=== Checking AUB Paper Counts on Semantic Scholar ===")
    logger.info("="*60)

    API_KEY = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"

    # Test different query variations
    queries = [
        "American University of Beirut",
        "American University of Beirut computer science",
        "American University of Beirut engineering",
        "American University of Beirut medicine",
        "AUB Beirut",
    ]

    logger.info("\nQuerying Semantic Scholar API...\n")

    total_counts = {}

    for query in queries:
        count = get_paper_count(query, API_KEY)
        total_counts[query] = count
        logger.info(f"'{query}'")
        logger.info(f"  → {count:,} papers\n")

    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    max_query = max(total_counts, key=total_counts.get)
    max_count = total_counts[max_query]

    logger.info(f"\nBest query: '{max_query}'")
    logger.info(f"Total papers: {max_count:,}")

    logger.info(f"\n⚠️  Note: Due to API 1000-result limit per query,")
    logger.info(f"   you'll need multiple queries to collect all {max_count:,} papers.")
    logger.info(f"\n   The comprehensive collection script uses 19 different queries")
    logger.info(f"   to maximize coverage while deduplicating automatically.")

    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
