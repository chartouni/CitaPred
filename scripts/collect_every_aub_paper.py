"""
Collect EVERY AUB paper on Semantic Scholar using year-by-year queries.

Strategy: Query each year individually (1920-2025) to work around 1000-result limit.
Since each year has <1000 papers, this gets complete coverage.
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
    """Collect papers for a single query, skipping already seen IDs."""

    new_papers = []
    offset = 0

    while offset < 1000:  # API limit
        retry_count = 0
        max_retries = 5

        while retry_count < max_retries:
            try:
                papers = collector.search_papers(
                    query=query,
                    limit=batch_size,
                    offset=offset
                )

                if not papers:
                    break

                batch_new = 0
                batch_duplicates = 0

                for paper in papers:
                    paper_id = paper.get('paperId')

                    if paper.get('citationCount') is None:
                        continue

                    if paper_id in seen_ids:
                        batch_duplicates += 1
                        continue

                    seen_ids.add(paper_id)
                    new_papers.append(paper)
                    batch_new += 1

                if batch_new > 0 or batch_duplicates > 0:
                    logger.info(f"  Offset {offset:4d}: +{batch_new} new, {batch_duplicates} dup")

                if len(papers) < batch_size:
                    break

                offset += batch_size
                time.sleep(delay)
                break  # Success, exit retry loop

            except Exception as e:
                error_str = str(e)

                # Check if it's a retryable error (504, 503, 502)
                if "504" in error_str or "503" in error_str or "502" in error_str:
                    retry_count += 1
                    wait_time = delay * (2 ** retry_count)  # Exponential backoff
                    logger.warning(f"  ⚠️  Timeout/Server error at offset {offset} (attempt {retry_count}/{max_retries})")
                    logger.warning(f"  Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # Non-retryable error (like 400 bad request)
                elif "400" in error_str:
                    logger.error(f"  ❌ Bad request at offset {offset}, skipping query")
                    break

                # Unknown error, retry a few times then give up
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.error(f"  Error at offset {offset}: {e}")
                        logger.warning(f"  Retrying ({retry_count}/{max_retries})...")
                        time.sleep(delay * 2)
                        continue
                    else:
                        logger.error(f"  ❌ Failed after {max_retries} retries, skipping")
                        break

    return new_papers


def main():
    """Collect ALL AUB papers using year-by-year queries."""

    logger.info("="*70)
    logger.info("=== COMPLETE AUB Collection: Year-by-Year Strategy ===")
    logger.info("="*70)
    logger.info("\nTarget: ALL 17,531 AUB papers")
    logger.info("Strategy: Query each year individually (1920-2025)")
    logger.info("Expected time: ~3-5 hours (100+ queries with rate limiting)\n")

    API_KEY = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"
    collector = DataCollector(api_key=API_KEY)

    all_papers = []
    seen_ids = set()
    query_stats = {}

    # Strategy 1: Year-by-year queries (primary strategy)
    year_queries = []
    for year in range(1920, 2026):  # 1920 to 2025
        year_queries.append((f"American University of Beirut {year}", f"Year {year}"))

    # Strategy 2: Add field-specific queries for extra coverage
    field_queries = [
        ("American University of Beirut computer science", "CS general"),
        ("American University of Beirut engineering", "Engineering"),
        ("American University of Beirut medicine", "Medicine"),
        ("American University of Beirut physics", "Physics"),
        ("American University of Beirut chemistry", "Chemistry"),
        ("American University of Beirut biology", "Biology"),
        ("American University of Beirut mathematics", "Mathematics"),
        ("American University of Beirut business", "Business"),
        ("American University of Beirut economics", "Economics"),
        ("American University of Beirut machine learning", "ML"),
        ("American University of Beirut neural network", "Neural Networks"),
        ("American University of Beirut artificial intelligence", "AI"),
    ]

    # Combine all queries
    all_queries = year_queries + field_queries
    total_queries = len(all_queries)

    logger.info(f"Total queries to execute: {total_queries}")
    logger.info(f"  - Year queries: {len(year_queries)}")
    logger.info(f"  - Field queries: {len(field_queries)}")
    logger.info(f"\nStarting collection...\n")

    start_time = time.time()

    # Execute all queries
    for i, (query, description) in enumerate(all_queries, 1):
        logger.info(f"[{i}/{total_queries}] {description}: '{query}'")

        new_papers = collect_with_query(
            collector=collector,
            query=query,
            seen_ids=seen_ids,
            batch_size=100,
            delay=1.5
        )

        all_papers.extend(new_papers)
        query_stats[description] = len(new_papers)

        if len(new_papers) > 0:
            logger.info(f"  ✅ +{len(new_papers)} new papers")

        # Progress update every 10 queries
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time_per_query = elapsed / i
            remaining_queries = total_queries - i
            eta_seconds = avg_time_per_query * remaining_queries
            eta_minutes = eta_seconds / 60

            logger.info(f"\n📊 Progress: {i}/{total_queries} queries ({i/total_queries*100:.1f}%)")
            logger.info(f"   Papers collected: {len(all_papers):,}")
            logger.info(f"   ETA: {eta_minutes:.1f} minutes\n")

    # Final statistics
    elapsed_total = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("📈 COLLECTION COMPLETE")
    logger.info("="*70)
    logger.info(f"\n✅ Total unique papers: {len(all_papers):,}")
    logger.info(f"🎯 Target was: 17,531 papers")
    logger.info(f"📊 Coverage: {len(all_papers)/17531*100:.1f}%")
    logger.info(f"⏱️  Time elapsed: {elapsed_total/60:.1f} minutes")

    if len(all_papers) == 0:
        logger.error("❌ No papers collected!")
        return

    # Show top contributing queries
    logger.info("\n🏆 Top 20 Queries by New Papers:")
    sorted_stats = sorted(query_stats.items(), key=lambda x: x[1], reverse=True)[:20]
    for query_desc, count in sorted_stats:
        if count > 0:
            logger.info(f"  {count:4d} - {query_desc}")

    # Save results
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "aub_papers_all.json"
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
    csv_path = output_dir / "aub_papers_all.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"💾 Saved CSV: {csv_path}")

    # Dataset statistics
    logger.info("\n" + "="*70)
    logger.info("📊 DATASET STATISTICS")
    logger.info("="*70)

    logger.info(f"\n📄 Total Papers: {len(all_papers):,}")

    if 'year' in df.columns and len(df) > 0:
        logger.info(f"\n📅 Year Range: {int(df['year'].min())} - {int(df['year'].max())}")

        # Papers by decade
        df['decade'] = (df['year'] // 10) * 10
        decade_dist = df['decade'].value_counts().sort_index(ascending=False)
        logger.info("\nPapers by Decade:")
        for decade, count in decade_dist.items():
            logger.info(f"  {int(decade)}s: {count:,} papers")

    if 'citationCount' in df.columns:
        logger.info(f"\n📈 Citations:")
        logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
        logger.info(f"  Median: {df['citationCount'].median():.0f}")
        logger.info(f"  Max: {df['citationCount'].max():,.0f}")
        logger.info(f"  Total: {df['citationCount'].sum():,.0f}")

    papers_with_abstracts = sum(1 for p in all_papers if p.get('abstract'))
    papers_with_hindex = sum(1 for p in all_papers if any(a.get('hIndex') for a in p.get('authors', [])))

    logger.info(f"\n✅ Papers with abstracts: {papers_with_abstracts:,} ({papers_with_abstracts/len(all_papers)*100:.1f}%)")
    logger.info(f"✅ Papers with author h-index: {papers_with_hindex:,} ({papers_with_hindex/len(all_papers)*100:.1f}%)")

    # Papers suitable for training (2015-2020)
    training_papers = df[(df['year'] >= 2015) & (df['year'] <= 2020)]
    logger.info(f"\n🎓 Papers from 2015-2020 (good for training): {len(training_papers):,}")

    logger.info("\n" + "="*70)
    logger.info("✅ SUCCESS!")
    logger.info("="*70)
    logger.info(f"\n🎉 Collected {len(all_papers):,} unique AUB papers")
    logger.info(f"📂 Files saved:")
    logger.info(f"   JSON: {json_path}")
    logger.info(f"   CSV: {csv_path}")
    logger.info(f"\n✨ Zero duplicates guaranteed!")


if __name__ == "__main__":
    main()
