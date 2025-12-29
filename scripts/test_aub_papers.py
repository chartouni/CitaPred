"""
Test script to check if AUB papers are available in Semantic Scholar.

This script tries different search strategies to find papers from
American University of Beirut researchers.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.data.collector import DataCollector
from citapred.utils.logger import setup_logger

logger = setup_logger()


def test_search_strategy(collector, query, description):
    """Test a search query and report results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {description}")
    logger.info(f"Query: '{query}'")
    logger.info(f"{'='*60}")

    try:
        papers = collector.search_papers(query=query, limit=10)

        if papers:
            logger.info(f"✅ Found {len(papers)} papers")
            logger.info("\nFirst 3 papers:")
            for i, paper in enumerate(papers[:3], 1):
                logger.info(f"\n{i}. {paper.get('title', 'N/A')}")
                logger.info(f"   Authors: {', '.join([a.get('name', 'N/A') for a in paper.get('authors', [])[:3]])}")
                logger.info(f"   Venue: {paper.get('venue', 'N/A')}")
                logger.info(f"   Year: {paper.get('year', 'N/A')}")
                logger.info(f"   Citations: {paper.get('citationCount', 'N/A')}")
        else:
            logger.warning("❌ No papers found")

        time.sleep(1)  # Rate limiting
        return len(papers)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 0


def main():
    """Test various strategies to find AUB papers."""

    logger.info("=== AUB Papers Availability Test ===\n")

    # Your Semantic Scholar API key
    API_KEY = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"

    collector = DataCollector(api_key=API_KEY)

    # Test different search strategies
    strategies = [
        # Strategy 1: Direct university name
        ("American University of Beirut", "Direct university name in search"),

        # Strategy 2: University abbreviation
        ("AUB", "University abbreviation"),

        # Strategy 3: Location + keywords
        ("Beirut machine learning", "Location + research topic"),

        # Strategy 4: Specific research topics that might be at AUB
        ("machine learning Lebanon", "Research topic + country"),

        # Strategy 5: Try searching for papers mentioning AUB in text
        ("university beirut neural networks", "University location + topic"),

        # Strategy 6: Computer science + location
        ("computer science beirut", "Field + location"),
    ]

    results = {}
    total_found = 0

    for query, description in strategies:
        count = test_search_strategy(collector, query, description)
        results[description] = count
        total_found += count

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    for strategy, count in results.items():
        status = "✅" if count > 0 else "❌"
        logger.info(f"{status} {strategy}: {count} papers")

    logger.info(f"\nTotal papers found across all strategies: {total_found}")

    if total_found == 0:
        logger.warning("\n⚠️  No AUB papers found via search queries.")
        logger.info("\n📋 RECOMMENDATION:")
        logger.info("   1. Ask AUB for list of paper titles or DOIs")
        logger.info("   2. Use Semantic Scholar API to look up each paper individually")
        logger.info("   3. This approach is more reliable than search queries")
        logger.info("\n   Example workflow:")
        logger.info("   - AUB provides: ['Paper Title 1', 'Paper Title 2', ...]")
        logger.info("   - Script looks up each title in Semantic Scholar")
        logger.info("   - Enriches with citation counts, h-index, etc.")
    else:
        logger.info("\n✅ Found papers! You can collect AUB data via search queries.")
        logger.info(f"   Most effective strategy: {max(results, key=results.get)}")

    logger.info("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
