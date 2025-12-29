"""
Collect ALL AUB papers from Semantic Scholar using multiple query strategies.

This script uses various query combinations to work around the 1000-result limit
while ensuring no duplicates through paperId tracking.
"""

import sys
from pathlib import Path
import time
import json
from typing import List, Dict, Set
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.data.collector import DataCollector
from citapred.utils.logger import setup_logger

logger = setup_logger()


def collect_with_query(
    collector: DataCollector,
    query: str,
    seen_ids: Set[str],
    batch_size: int = 100,
    delay: float = 1.5
) -> List[Dict]:
    """
    Collect papers for a single query, skipping already seen IDs.

    Args:
        collector: DataCollector instance
        query: Search query
        seen_ids: Set of already collected paper IDs
        batch_size: Papers per request
        delay: Delay between requests

    Returns:
        List of new (unseen) papers
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Query: '{query}'")
    logger.info(f"{'='*60}")

    new_papers = []
    offset = 0

    while offset < 1000:  # API limit
        try:
            papers = collector.search_papers(
                query=query,
                limit=batch_size,
                offset=offset
            )

            if not papers:
                break

            # Filter for papers with citations AND not already seen
            batch_new = 0
            batch_duplicates = 0

            for paper in papers:
                paper_id = paper.get('paperId')

                # Skip if no citation data or already seen
                if paper.get('citationCount') is None:
                    continue

                if paper_id in seen_ids:
                    batch_duplicates += 1
                    continue

                # New paper!
                seen_ids.add(paper_id)
                new_papers.append(paper)
                batch_new += 1

            logger.info(f"Offset {offset}: {batch_new} new, {batch_duplicates} duplicates")

            if len(papers) < batch_size:  # No more results
                break

            offset += batch_size
            time.sleep(delay)

        except Exception as e:
            logger.error(f"Error at offset {offset}: {e}")
            if "400" in str(e):
                break
            time.sleep(delay * 2)
            continue

    logger.info(f"✅ Total new papers from this query: {len(new_papers)}")
    return new_papers


def main():
    """Collect ALL AUB papers using multiple query strategies."""

    logger.info("="*60)
    logger.info("=== Comprehensive AUB Papers Collection ===")
    logger.info("="*60)
    logger.info("\nStrategy: Multiple queries with automatic deduplication\n")

    API_KEY = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"
    collector = DataCollector(api_key=API_KEY)

    # Multiple query strategies to maximize coverage
    queries = [
        # Strategy 1: Base query
        "American University of Beirut",

        # Strategy 2: By major field
        "American University of Beirut computer science",
        "American University of Beirut engineering",
        "American University of Beirut medicine",
        "American University of Beirut physics",
        "American University of Beirut chemistry",
        "American University of Beirut biology",
        "American University of Beirut mathematics",
        "American University of Beirut business",
        "American University of Beirut economics",
        "American University of Beirut psychology",

        # Strategy 3: By location variations
        "AUB Beirut",
        "Beirut Lebanon university research",

        # Strategy 4: By specific departments/topics (CS-focused)
        "American University of Beirut machine learning",
        "American University of Beirut artificial intelligence",
        "American University of Beirut neural network",
        "American University of Beirut data science",
        "American University of Beirut software",
        "American University of Beirut algorithm",
    ]

    all_papers = []
    seen_ids = set()  # Track paper IDs to prevent duplicates
    query_stats = {}

    logger.info(f"Total queries to execute: {len(queries)}\n")

    # Collect papers from each query
    for i, query in enumerate(queries, 1):
        logger.info(f"\n[{i}/{len(queries)}] Processing query...")

        new_papers = collect_with_query(
            collector=collector,
            query=query,
            seen_ids=seen_ids,
            batch_size=100,
            delay=1.5
        )

        all_papers.extend(new_papers)
        query_stats[query] = len(new_papers)

        logger.info(f"📊 Running total: {len(all_papers)} unique papers")

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("📈 COLLECTION SUMMARY")
    logger.info("="*60)
    logger.info(f"\n✅ Total unique papers collected: {len(all_papers)}")
    logger.info(f"🔍 Queries executed: {len(queries)}")
    logger.info(f"🎯 Deduplication: {len(seen_ids)} unique IDs tracked")

    # Show contribution by query
    logger.info("\n📊 New Papers Contributed by Each Query:")
    sorted_stats = sorted(query_stats.items(), key=lambda x: x[1], reverse=True)
    for query, count in sorted_stats:
        if count > 0:
            logger.info(f"  {count:4d} papers - {query}")

    if len(all_papers) == 0:
        logger.error("❌ No papers collected!")
        return

    # Save results
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "aub_papers_complete.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    logger.info(f"\n💾 Saved JSON: {json_path}")

    # Save CSV
    csv_data = []
    for paper in all_papers:
        row = {
            'paperId': paper.get('paperId'),
            'title': paper.get('title'),
            'year': paper.get('year'),
            'venue': paper.get('venue'),
            'citationCount': paper.get('citationCount'),
            'referenceCount': paper.get('referenceCount'),
            'abstract': (paper.get('abstract', '')[:200] + '...') if paper.get('abstract') else '',
            'author_count': len(paper.get('authors', [])),
            'authors': ', '.join([a.get('name', 'N/A') for a in paper.get('authors', [])[:5]]),
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
    csv_path = output_dir / "aub_papers_complete.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"💾 Saved CSV: {csv_path}")

    # Statistics
    logger.info("\n" + "="*60)
    logger.info("📊 DATASET STATISTICS")
    logger.info("="*60)

    logger.info(f"\n📄 Total Papers: {len(all_papers)}")

    if 'year' in df.columns:
        logger.info(f"\n📅 Year Range: {df['year'].min():.0f} - {df['year'].max():.0f}")
        year_dist = df['year'].value_counts().sort_index(ascending=False).head(10)
        logger.info("\nTop 10 Years:")
        for year, count in year_dist.items():
            logger.info(f"  {int(year)}: {count} papers")

    if 'citationCount' in df.columns:
        logger.info(f"\n📈 Citations:")
        logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
        logger.info(f"  Median: {df['citationCount'].median():.0f}")
        logger.info(f"  Max: {df['citationCount'].max():.0f}")

    # Papers with good data for training
    recent_papers = df[(df['year'] >= 2015) & (df['year'] <= 2020)]
    papers_with_abstracts = sum(1 for p in all_papers if p.get('abstract'))
    papers_with_hindex = sum(1 for p in all_papers if any(a.get('hIndex') for a in p.get('authors', [])))

    logger.info(f"\n✅ Papers from 2015-2020: {len(recent_papers)} ({len(recent_papers)/len(df)*100:.1f}%)")
    logger.info(f"✅ Papers with abstracts: {papers_with_abstracts} ({papers_with_abstracts/len(all_papers)*100:.1f}%)")
    logger.info(f"✅ Papers with author h-index: {papers_with_hindex} ({papers_with_hindex/len(all_papers)*100:.1f}%)")

    logger.info("\n" + "="*60)
    logger.info("✅ COLLECTION COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nFiles saved:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"\n✨ {len(all_papers)} unique AUB papers collected with ZERO duplicates!")


if __name__ == "__main__":
    main()
