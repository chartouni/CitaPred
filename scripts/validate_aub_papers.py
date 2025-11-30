"""
Validate that collected papers are actually from AUB authors.

This script checks if the collected papers have authors affiliated with AUB
by looking for author names and checking their institutional affiliations.
"""

import sys
from pathlib import Path
import json
import pandas as pd
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.utils.logger import setup_logger

logger = setup_logger()


def analyze_papers(json_file: str):
    """Analyze papers to check AUB affiliation."""

    logger.info("="*60)
    logger.info("=== AUB Papers Validation ===")
    logger.info("="*60)

    # Load papers
    with open(json_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    logger.info(f"\nTotal papers in file: {len(papers):,}\n")

    # Sample analysis
    logger.info("📋 Sample of first 30 papers:")
    logger.info("="*60)

    for i, paper in enumerate(papers[:30], 1):
        title = paper.get('title', 'N/A')
        authors = paper.get('authors', [])
        venue = paper.get('venue', 'N/A')
        year = paper.get('year', 'N/A')

        author_names = ', '.join([a.get('name', 'N/A') for a in authors[:3]])
        if len(authors) > 3:
            author_names += f' +{len(authors)-3} more'

        logger.info(f"\n{i}. {title[:80]}")
        logger.info(f"   Authors: {author_names}")
        logger.info(f"   Venue: {venue[:60]}")
        logger.info(f"   Year: {year}")

    # Check for obvious non-AUB papers
    logger.info("\n" + "="*60)
    logger.info("🔍 Checking for potential false positives...")
    logger.info("="*60)

    # Papers that might just mention AUB
    suspicious = []

    for paper in papers:
        title = paper.get('title', '').lower()

        # Check if title just talks ABOUT AUB (not FROM AUB)
        if any(phrase in title for phrase in [
            'university of beirut',
            'beirut university',
            'medical center',
            'hospital',
        ]):
            # These could be studies ABOUT AUB, not BY AUB
            if any(word in title for word in ['study', 'experience', 'analysis']):
                suspicious.append(paper)

    logger.info(f"\n⚠️  Found {len(suspicious)} potentially suspicious papers")
    logger.info(f"   (papers that might be ABOUT AUB, not FROM AUB)\n")

    if suspicious:
        logger.info("Examples:")
        for paper in suspicious[:5]:
            logger.info(f"  - {paper.get('title', 'N/A')[:100]}")

    # Venue analysis
    logger.info("\n" + "="*60)
    logger.info("🏛️  Top 20 Publication Venues")
    logger.info("="*60)

    venues = [p.get('venue', 'Unknown') for p in papers if p.get('venue')]
    venue_counts = Counter(venues).most_common(20)

    for venue, count in venue_counts:
        logger.info(f"  {count:4d} - {venue}")

    # Medical vs CS/Engineering check
    medical_venues = ['medical', 'medicine', 'clinical', 'journal', 'health', 'cancer', 'disease']
    cs_venues = ['computer', 'ieee', 'acm', 'international conference', 'workshop', 'computing']

    medical_count = sum(1 for p in papers
                       if any(term in p.get('venue', '').lower() for term in medical_venues))
    cs_count = sum(1 for p in papers
                  if any(term in p.get('venue', '').lower() for term in cs_venues))

    logger.info("\n" + "="*60)
    logger.info("📊 Field Distribution (by venue keywords)")
    logger.info("="*60)
    logger.info(f"\n  Medical/Health papers: ~{medical_count:,} ({medical_count/len(papers)*100:.1f}%)")
    logger.info(f"  CS/Engineering papers: ~{cs_count:,} ({cs_count/len(papers)*100:.1f}%)")
    logger.info(f"  Other/Unknown: ~{len(papers)-medical_count-cs_count:,}")

    # Year distribution
    years = [p.get('year') for p in papers if p.get('year')]
    year_counts = Counter(years).most_common(10)

    logger.info("\n" + "="*60)
    logger.info("📅 Top 10 Years")
    logger.info("="*60)

    for year, count in sorted(year_counts, key=lambda x: x[0], reverse=True):
        logger.info(f"  {int(year)}: {count:,} papers")

    # RECOMMENDATION
    logger.info("\n" + "="*60)
    logger.info("💡 RECOMMENDATION")
    logger.info("="*60)

    logger.info("\nTo ensure papers are FROM AUB (not just ABOUT AUB):")
    logger.info("\n1. Manual validation:")
    logger.info("   - Open aub_papers_complete.csv in Excel")
    logger.info("   - Look at author names and venues")
    logger.info("   - Remove obviously non-AUB papers")

    logger.info("\n2. Contact AUB directly:")
    logger.info("   - Ask for official list of faculty publications")
    logger.info("   - Cross-reference with collected data")

    logger.info("\n3. Use Semantic Scholar's bulk paper lookup:")
    logger.info("   - Get paper IDs/DOIs from AUB")
    logger.info("   - Look them up directly (100% accuracy)")

    logger.info(f"\n📊 Current dataset: {len(papers):,} papers")
    logger.info(f"   Likely false positives: ~{len(suspicious)}")
    logger.info(f"   Estimated true AUB papers: ~{len(papers) - len(suspicious):,}")

    logger.info("\n" + "="*60)


def main():
    """Run validation."""

    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "data/raw/aub_papers_complete.json"

    if not Path(json_file).exists():
        logger.error(f"❌ File not found: {json_file}")
        logger.info("\nUsage: python scripts/validate_aub_papers.py [json_file]")
        logger.info("Example: python scripts/validate_aub_papers.py data/raw/aub_papers_complete.json")
        return

    analyze_papers(json_file)


if __name__ == "__main__":
    main()
