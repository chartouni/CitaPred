"""
Analyze venues in the dataset to identify missing prestige scores.

This script helps identify which venues need prestige scores added
to improve model performance.
"""

import sys
from pathlib import Path
import json
import pandas as pd
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.features.extractor import FeatureExtractor
from citapred.utils.logger import setup_logger

logger = setup_logger()


def analyze_venues(data_file: str = "data/raw/complete_dataset.json"):
    """Analyze venue distribution and identify missing prestige scores."""

    logger.info("=== CitaPred Venue Analysis ===\n")

    # Load data
    if not Path(data_file).exists():
        logger.error(f"Dataset not found: {data_file}")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} papers")

    # Extract venues
    venues = [p.get('venue', '') for p in papers if p.get('venue')]
    venue_counts = Counter(venues)

    logger.info(f"Unique venues: {len(venue_counts)}")
    logger.info(f"Total papers with venue: {sum(venue_counts.values())}\n")

    # Get prestige scores from feature extractor
    extractor = FeatureExtractor()
    known_venues = set(extractor.venue_prestige.keys())

    # Categorize venues
    venues_with_scores = []
    venues_without_scores = []

    for venue, count in venue_counts.most_common():
        # Check if venue has exact or partial match
        has_score = venue in known_venues
        if not has_score:
            # Check partial match
            venue_lower = venue.lower()
            for known_venue in known_venues:
                if known_venue.lower() in venue_lower or venue_lower in known_venue.lower():
                    has_score = True
                    break

        if has_score:
            venues_with_scores.append((venue, count))
        else:
            venues_without_scores.append((venue, count))

    # Calculate coverage
    papers_with_scores = sum(count for _, count in venues_with_scores)
    papers_without_scores = sum(count for _, count in venues_without_scores)
    total = papers_with_scores + papers_without_scores

    coverage_pct = (papers_with_scores / total * 100) if total > 0 else 0

    logger.info("="*60)
    logger.info("COVERAGE SUMMARY")
    logger.info("="*60)
    logger.info(f"Papers with venue scores: {papers_with_scores}/{total} ({coverage_pct:.1f}%)")
    logger.info(f"Papers without venue scores: {papers_without_scores}/{total} ({100-coverage_pct:.1f}%)")
    logger.info(f"Venues with scores: {len(venues_with_scores)}")
    logger.info(f"Venues without scores: {len(venues_without_scores)}\n")

    # Show top venues WITH scores
    logger.info("="*60)
    logger.info("TOP 10 VENUES WITH PRESTIGE SCORES")
    logger.info("="*60)
    for i, (venue, count) in enumerate(venues_with_scores[:10], 1):
        # Get actual score
        if venue in extractor.venue_prestige:
            score = extractor.venue_prestige[venue]
        else:
            # Find partial match
            score = 3.0  # default
            venue_lower = venue.lower()
            for known_venue, s in extractor.venue_prestige.items():
                if known_venue.lower() in venue_lower or venue_lower in known_venue.lower():
                    score = s
                    break

        logger.info(f"{i:2d}. {venue:40s} | Papers: {count:4d} | Score: {score:.1f}")

    # Show top venues WITHOUT scores (these need attention!)
    if venues_without_scores:
        logger.info("\n" + "="*60)
        logger.info("TOP 10 VENUES MISSING PRESTIGE SCORES ⚠️")
        logger.info("="*60)
        logger.info("These venues should be added to improve model performance:\n")

        for i, (venue, count) in enumerate(venues_without_scores[:10], 1):
            logger.info(f"{i:2d}. {venue:40s} | Papers: {count:4d}")

        # Calculate potential impact
        top_10_missing = sum(count for _, count in venues_without_scores[:10])
        impact_pct = (top_10_missing / total * 100) if total > 0 else 0

        logger.info(f"\n📊 Top 10 missing venues affect {top_10_missing} papers ({impact_pct:.1f}% of dataset)")

        # Generate code snippet to add
        logger.info("\n" + "="*60)
        logger.info("SUGGESTED CODE TO ADD")
        logger.info("="*60)
        logger.info("\nAdd these to src/citapred/features/extractor.py in self.venue_prestige:\n")

        for venue, count in venues_without_scores[:10]:
            # Estimate score based on paper count (rough heuristic)
            if count > 50:
                estimated_score = 7.5
            elif count > 20:
                estimated_score = 6.5
            else:
                estimated_score = 5.0

            logger.info(f"    '{venue}': {estimated_score},  # {count} papers")

    # Show venue citation statistics
    logger.info("\n" + "="*60)
    logger.info("VENUE CITATION STATISTICS")
    logger.info("="*60)

    # Calculate stats per venue
    df = pd.DataFrame(papers)
    if 'venue' in df.columns and 'citationCount' in df.columns:
        venue_stats = df.groupby('venue')['citationCount'].agg(['mean', 'median', 'count', 'max'])
        venue_stats = venue_stats.sort_values('mean', ascending=False)

        logger.info("\nTop 10 venues by average citations:\n")
        logger.info(f"{'Venue':<40s} | {'Mean':>8s} | {'Median':>8s} | {'Max':>8s} | {'Papers':>7s}")
        logger.info("-" * 85)

        for venue, row in venue_stats.head(10).iterrows():
            logger.info(f"{venue:<40s} | {row['mean']:8.0f} | {row['median']:8.0f} | {row['max']:8.0f} | {int(row['count']):7d}")

    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)

    if venues_without_scores:
        logger.info("\n1. Add prestige scores for top missing venues (see code above)")
        logger.info(f"   Expected R² improvement: +{len(venues_without_scores[:10]) * 0.002:.3f} to +{len(venues_without_scores[:10]) * 0.005:.3f}")

    if coverage_pct < 90:
        logger.info("\n2. Current coverage is {:.1f}% - aim for 90%+ for best results".format(coverage_pct))
    else:
        logger.info("\n✅ Good coverage! ({:.1f}% of papers have venue scores)".format(coverage_pct))

    logger.info("\n3. Verify scores are appropriate for each venue's impact factor")
    logger.info("   - Top tier (Nature, Science): 9-10")
    logger.info("   - Top conferences (NeurIPS, CVPR): 8-9")
    logger.info("   - Good venues: 6-8")
    logger.info("   - Standard venues: 4-6")

    logger.info("\n=== Analysis Complete ===\n")


if __name__ == "__main__":
    analyze_venues()
