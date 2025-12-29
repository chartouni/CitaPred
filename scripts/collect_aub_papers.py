"""
Collect all AUB papers from Semantic Scholar API.

This script fetches all available papers affiliated with American University of Beirut
and saves them to both CSV (for viewing) and JSON (for training).
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


def collect_all_aub_papers(
    api_key: str,
    batch_size: int = 100,
    delay: float = 1.5
) -> List[Dict]:
    """
    Collect all AUB papers from Semantic Scholar.

    Args:
        api_key: Semantic Scholar API key
        batch_size: Papers per API request
        delay: Delay between requests (seconds)

    Returns:
        List of paper dictionaries
    """
    collector = DataCollector(api_key=api_key)
    all_papers = []

    # Try different query strategies
    queries = [
        "American University of Beirut",
        # Could add more specific queries if needed
        # "American University of Beirut computer science",
        # "American University of Beirut engineering",
    ]

    for query in queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting papers for: {query}")
        logger.info(f"{'='*60}")

        offset = 0
        query_papers = []

        while True:
            logger.info(f"Progress: {len(query_papers)} papers collected (offset: {offset})")

            try:
                # Fetch batch with offset for pagination
                papers = collector.search_papers(
                    query=query,
                    limit=batch_size,
                    offset=offset
                )

                if not papers:
                    logger.info("No more papers found for this query")
                    break

                # Filter for papers with citation data
                filtered_papers = []
                for paper in papers:
                    # Check if paper has required fields
                    if paper.get('citationCount') is not None:
                        filtered_papers.append(paper)

                logger.info(f"Found {len(filtered_papers)} valid papers in this batch")
                query_papers.extend(filtered_papers)

                # Check if we hit the 1000-offset API limit
                if offset >= 900:  # Stop before hitting 1000 limit
                    logger.warning(f"Reached API offset limit (1000) for query: {query}")
                    break

                # Rate limiting
                time.sleep(delay)
                offset += batch_size

            except Exception as e:
                logger.error(f"Error collecting batch at offset {offset}: {e}")
                if "400" in str(e):  # API offset limit error
                    logger.warning("Hit API offset limit")
                    break
                time.sleep(delay * 2)  # Wait longer on error
                continue

        logger.info(f"Total papers collected for '{query}': {len(query_papers)}")
        all_papers.extend(query_papers)

    return all_papers


def save_papers(papers: List[Dict], output_dir: str = "data/raw"):
    """
    Save collected papers to CSV and JSON.

    Args:
        papers: List of paper dictionaries
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Remove duplicates based on paperId
    logger.info(f"\nRemoving duplicates from {len(papers)} papers...")
    unique_papers = []
    seen_ids = set()

    for paper in papers:
        paper_id = paper.get('paperId')
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(paper)

    logger.info(f"Unique papers: {len(unique_papers)}")

    # Save as JSON (for training scripts)
    json_path = output_path / "aub_papers.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(unique_papers, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Saved JSON: {json_path}")

    # Save as CSV (for easy viewing/filtering)
    csv_path = output_path / "aub_papers.csv"

    # Flatten author data for CSV
    csv_data = []
    for paper in unique_papers:
        row = {
            'paperId': paper.get('paperId'),
            'title': paper.get('title'),
            'year': paper.get('year'),
            'venue': paper.get('venue'),
            'citationCount': paper.get('citationCount'),
            'referenceCount': paper.get('referenceCount'),
            'abstract': paper.get('abstract', '')[:200] + '...' if paper.get('abstract') else '',  # Truncate for CSV
            'author_count': len(paper.get('authors', [])),
            'authors': ', '.join([a.get('name', 'N/A') for a in paper.get('authors', [])[:5]]),  # First 5 authors
        }

        # Add author h-index stats
        authors = paper.get('authors', [])
        h_indices = [a.get('hIndex') for a in authors if a.get('hIndex') is not None]
        if h_indices:
            row['max_author_hindex'] = max(h_indices)
            row['mean_author_hindex'] = sum(h_indices) / len(h_indices)
        else:
            row['max_author_hindex'] = None
            row['mean_author_hindex'] = None

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"✅ Saved CSV: {csv_path}")

    return unique_papers, df


def print_statistics(papers: List[Dict], df: pd.DataFrame):
    """Print statistics about collected papers."""
    logger.info("\n" + "="*60)
    logger.info("📊 AUB Papers Statistics")
    logger.info("="*60)

    logger.info(f"\n📄 Total Papers: {len(papers)}")

    # Year distribution
    if 'year' in df.columns:
        logger.info(f"\n📅 Year Range: {df['year'].min()} - {df['year'].max()}")
        year_dist = df['year'].value_counts().sort_index(ascending=False).head(10)
        logger.info("\nTop 10 Years by Paper Count:")
        for year, count in year_dist.items():
            logger.info(f"  {year}: {count} papers")

    # Citation statistics
    if 'citationCount' in df.columns:
        logger.info(f"\n📈 Citation Statistics:")
        logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
        logger.info(f"  Median: {df['citationCount'].median():.0f}")
        logger.info(f"  Min: {df['citationCount'].min()}")
        logger.info(f"  Max: {df['citationCount'].max()}")
        logger.info(f"  Total citations: {df['citationCount'].sum():.0f}")

    # Venue distribution
    if 'venue' in df.columns:
        venue_dist = df[df['venue'].notna()]['venue'].value_counts().head(10)
        logger.info(f"\n🏛️  Top 10 Venues:")
        for venue, count in venue_dist.items():
            logger.info(f"  {venue}: {count} papers")

    # Papers with abstracts
    abstracts_count = sum(1 for p in papers if p.get('abstract'))
    logger.info(f"\n📝 Papers with abstracts: {abstracts_count} ({abstracts_count/len(papers)*100:.1f}%)")

    # Papers with author h-index
    h_index_count = sum(1 for p in papers if any(a.get('hIndex') for a in p.get('authors', [])))
    logger.info(f"👤 Papers with author h-index: {h_index_count} ({h_index_count/len(papers)*100:.1f}%)")

    # Most cited papers
    logger.info("\n🌟 Top 10 Most Cited Papers:")
    top_cited = df.nlargest(10, 'citationCount')[['title', 'year', 'citationCount', 'venue']]
    for idx, row in top_cited.iterrows():
        logger.info(f"\n  {row['citationCount']} citations - {row['title'][:80]}")
        logger.info(f"    Year: {row['year']}, Venue: {row['venue']}")


def main():
    """Main collection pipeline for AUB papers."""

    logger.info("=== AUB Papers Collection Script ===\n")

    # Configuration
    API_KEY = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"
    BATCH_SIZE = 100
    DELAY = 1.5  # Seconds between requests

    # Collect papers
    logger.info("Starting collection...")
    papers = collect_all_aub_papers(
        api_key=API_KEY,
        batch_size=BATCH_SIZE,
        delay=DELAY
    )

    if not papers:
        logger.error("❌ No papers collected!")
        return

    # Save papers
    unique_papers, df = save_papers(papers, output_dir="data/raw")

    # Print statistics
    print_statistics(unique_papers, df)

    logger.info("\n" + "="*60)
    logger.info("✅ Collection Complete!")
    logger.info("="*60)
    logger.info(f"\nFiles saved:")
    logger.info(f"  - JSON: data/raw/aub_papers.json ({len(unique_papers)} papers)")
    logger.info(f"  - CSV: data/raw/aub_papers.csv ({len(unique_papers)} papers)")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review CSV file to inspect data quality")
    logger.info(f"  2. Filter papers by year range if needed (e.g., 2015-2020)")
    logger.info(f"  3. Use JSON file with training scripts")


if __name__ == "__main__":
    main()
