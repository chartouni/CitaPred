"""
Script to collect a large dataset of research papers from Semantic Scholar.

This script collects papers with proper rate limiting, saves progress incrementally,
and filters for papers with sufficient metadata.
"""

import sys
from pathlib import Path
import time
import json
from typing import List, Dict
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.data.collector import DataCollector
from citapred.utils.logger import setup_logger

logger = setup_logger()


def collect_papers_by_query(
    query: str,
    target_count: int = 2000,
    min_year: int = 2015,
    max_year: int = 2020,
    batch_size: int = 100,
    delay: float = 1.0,
    api_key: str = None
) -> List[Dict]:
    """
    Collect papers matching a query with rate limiting.

    Args:
        query: Search query (e.g., "machine learning", "deep learning")
        target_count: Target number of papers to collect
        min_year: Minimum publication year
        max_year: Maximum publication year (to ensure citation history)
        batch_size: Papers per API request
        delay: Delay between requests (seconds)
        api_key: Semantic Scholar API key

    Returns:
        List of paper dictionaries
    """
    collector = DataCollector(api_key=api_key)
    all_papers = []

    logger.info(f"Collecting papers for query: '{query}'")
    logger.info(f"Target: {target_count} papers from {min_year}-{max_year}")

    # Collect in batches
    offset = 0
    while len(all_papers) < target_count:
        logger.info(f"Progress: {len(all_papers)}/{target_count} papers collected")

        try:
            # Fetch batch with offset for pagination
            papers = collector.search_papers(query=query, limit=batch_size, offset=offset)

            if not papers:
                logger.warning("No more papers found")
                break

            # Filter by year and quality
            filtered_papers = []
            for paper in papers:
                # Check if paper has required fields
                if not paper.get('year') or not paper.get('title'):
                    continue

                year = paper.get('year', 0)
                if min_year <= year <= max_year:
                    # Check if paper has citation count
                    if paper.get('citationCount') is not None:
                        filtered_papers.append(paper)

            logger.info(f"Found {len(filtered_papers)} valid papers in this batch (offset: {offset})")
            all_papers.extend(filtered_papers)

            # Rate limiting
            time.sleep(delay)
            offset += batch_size

        except Exception as e:
            logger.error(f"Error collecting batch: {e}")
            time.sleep(delay * 2)  # Wait longer on error
            continue

    logger.info(f"Collection complete: {len(all_papers)} papers collected")
    return all_papers[:target_count]


def collect_papers_by_venue(
    venue: str,
    target_count: int = 2000,
    min_year: int = 2015,
    max_year: int = 2020,
    delay: float = 1.0
) -> List[Dict]:
    """
    Collect papers from a specific venue.

    Args:
        venue: Venue name (e.g., "NeurIPS", "Nature", "ICML")
        target_count: Target number of papers
        min_year: Minimum publication year
        max_year: Maximum publication year
        delay: Delay between requests

    Returns:
        List of paper dictionaries
    """
    query = f"venue:{venue}"
    return collect_papers_by_query(
        query=query,
        target_count=target_count,
        min_year=min_year,
        max_year=max_year,
        delay=delay
    )


def save_dataset(papers: List[Dict], filename: str, data_dir: str = "data/raw"):
    """
    Save collected papers to file.

    Args:
        papers: List of paper dictionaries
        filename: Output filename
        data_dir: Output directory
    """
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(papers)} papers to {output_path}")

    # Also save as CSV for easy viewing
    df = pd.DataFrame(papers)
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Also saved as CSV: {csv_path}")


def main():
    """Main data collection pipeline."""

    logger.info("=== CitaPred Data Collection Script ===")

    # Configuration
    API_KEY = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"  # Semantic Scholar API key

    QUERIES = [
        "machine learning",
        "deep learning",
        "natural language processing",
        "computer vision",
    ]

    PAPERS_PER_QUERY = 2000  # Total target: ~7000+ unique papers after deduplication
    MIN_YEAR = 2015
    MAX_YEAR = 2020  # Papers up to 2020, so they have 3+ years of citations
    DELAY = 1.5  # 1.5 seconds between requests to be safe with rate limit

    all_papers = []

    # Collect papers for each query
    for query in QUERIES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting papers for: {query}")
        logger.info(f"{'='*60}")

        papers = collect_papers_by_query(
            query=query,
            target_count=PAPERS_PER_QUERY,
            min_year=MIN_YEAR,
            max_year=MAX_YEAR,
            delay=DELAY,
            api_key=API_KEY
        )

        all_papers.extend(papers)

        # Save intermediate results
        save_dataset(
            papers,
            filename=f"papers_{query.replace(' ', '_')}.json",
            data_dir="data/raw"
        )

    # Remove duplicates (papers may appear in multiple queries)
    logger.info(f"\nRemoving duplicates from {len(all_papers)} papers...")
    unique_papers = []
    seen_ids = set()

    for paper in all_papers:
        paper_id = paper.get('paperId')
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(paper)

    logger.info(f"Unique papers: {len(unique_papers)}")

    # Save complete dataset
    save_dataset(
        unique_papers,
        filename="complete_dataset.json",
        data_dir="data/raw"
    )

    # Print statistics
    df = pd.DataFrame(unique_papers)
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total papers: {len(df)}")
    logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
    logger.info(f"Citation statistics:")
    logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
    logger.info(f"  Median: {df['citationCount'].median():.2f}")
    logger.info(f"  Min: {df['citationCount'].min()}")
    logger.info(f"  Max: {df['citationCount'].max()}")
    logger.info(f"Papers with abstracts: {df['abstract'].notna().sum()} ({df['abstract'].notna().sum()/len(df)*100:.1f}%)")

    logger.info("\n=== Collection Complete! ===")
    logger.info(f"Data saved to: data/raw/complete_dataset.json")


if __name__ == "__main__":
    main()
