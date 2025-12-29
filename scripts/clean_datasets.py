"""
Clean collected papers by removing invalid entries.

This script filters out:
- Papers with corrupted/garbled titles (encoding issues)
- Papers with missing critical fields
- Papers with non-English titles (likely false positives)
- Papers with suspiciously low data quality
"""

import sys
from pathlib import Path
import json
import re
from typing import List, Dict
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.utils.logger import setup_logger

logger = setup_logger()


def is_garbled_text(text: str) -> bool:
    """Check if text contains garbled/corrupted characters."""
    if not text:
        return False

    # Check for common garbled patterns
    garbled_patterns = [
        r'[Ø-Û]{3,}',  # Multiple Arabic/Persian characters in a row
        r'[À-ÿ]{5,}',  # Multiple special characters
        r'â€',  # Common encoding corruption
        r'Ã',   # Another common corruption
        r'\?{3,}',  # Multiple question marks (failed decode)
    ]

    for pattern in garbled_patterns:
        if re.search(pattern, text):
            return True

    # Check if text is mostly non-ASCII
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and (non_ascii_count / len(text)) > 0.3:
        return True

    return False


def is_english_text(text: str) -> bool:
    """Check if text is primarily English."""
    if not text:
        return False

    # Check for basic English patterns
    english_words = ['the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for']
    text_lower = text.lower()

    # Count English indicator words
    english_count = sum(1 for word in english_words if word in text_lower)

    # Check ASCII ratio
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_count / len(text) if len(text) > 0 else 0

    return english_count >= 2 or ascii_ratio > 0.9


def is_valid_paper(paper: Dict) -> tuple[bool, str]:
    """
    Check if paper is valid for training.

    Returns:
        (is_valid, reason)
    """
    # Check for required fields
    if not paper.get('title'):
        return False, "Missing title"

    if paper.get('citationCount') is None:
        return False, "Missing citation count"

    # Check for garbled title
    title = paper.get('title', '')
    if is_garbled_text(title):
        return False, f"Garbled title: {title[:50]}"

    # Check if title is English
    if not is_english_text(title):
        return False, f"Non-English title: {title[:50]}"

    # Check title length (too short or too long is suspicious)
    if len(title) < 10:
        return False, "Title too short"

    if len(title) > 500:
        return False, "Title too long (likely corrupted)"

    # Check year validity
    year = paper.get('year')
    if year and (year < 1900 or year > 2025):
        return False, f"Invalid year: {year}"

    # Check authors
    authors = paper.get('authors', [])
    if not authors or len(authors) == 0:
        return False, "No authors"

    # Check for garbled author names
    for author in authors:
        author_name = author.get('name', '')
        if author_name:
            # Check if author name is garbled
            if is_garbled_text(author_name):
                return False, f"Garbled author name: {author_name[:30]}"

            # Check for obviously corrupted names (too many caps, weird patterns)
            if author_name.isupper() and len(author_name) > 10:
                # All caps names like "BOOKS RECEIVED" are suspicious
                return False, f"Suspicious author name (all caps): {author_name[:30]}"

            # Check for numeric-heavy names (corrupted data)
            digit_count = sum(1 for c in author_name if c.isdigit())
            if len(author_name) > 0 and (digit_count / len(author_name)) > 0.3:
                return False, f"Author name has too many digits: {author_name[:30]}"

    # All checks passed
    return True, "Valid"


def clean_dataset(input_file: str, output_file: str):
    """Clean a dataset JSON file."""

    logger.info("="*60)
    logger.info("=== Data Cleaning ===")
    logger.info("="*60)

    # Load papers
    logger.info(f"\nLoading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    logger.info(f"Total papers loaded: {len(papers):,}")

    # Filter papers
    valid_papers = []
    invalid_reasons = {}

    for paper in papers:
        is_valid, reason = is_valid_paper(paper)

        if is_valid:
            valid_papers.append(paper)
        else:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    # Statistics
    logger.info(f"\n✅ Valid papers: {len(valid_papers):,} ({len(valid_papers)/len(papers)*100:.1f}%)")
    logger.info(f"❌ Invalid papers: {len(papers) - len(valid_papers):,}")

    if invalid_reasons:
        logger.info("\nReasons for removal:")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {count:4d} - {reason}")

    # Save cleaned dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(valid_papers, f, indent=2, ensure_ascii=False)

    logger.info(f"\n💾 Saved cleaned dataset: {output_path}")

    # Also save CSV
    csv_path = output_path.with_suffix('.csv')
    csv_data = []

    # Track h-index stats for debugging
    papers_with_hindex_csv = 0

    for paper in valid_papers:
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
            papers_with_hindex_csv += 1
        else:
            row['max_author_hindex'] = None
            row['mean_author_hindex'] = None

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"💾 Saved cleaned CSV: {csv_path}")
    logger.info(f"   Papers with h-index in CSV: {papers_with_hindex_csv}/{len(valid_papers)} ({papers_with_hindex_csv/len(valid_papers)*100:.1f}%)")

    # Final statistics
    logger.info("\n" + "="*60)
    logger.info("📊 Cleaned Dataset Statistics")
    logger.info("="*60)

    logger.info(f"\n📄 Total papers: {len(valid_papers):,}")

    if 'year' in df.columns and len(df) > 0:
        logger.info(f"\n📅 Year range: {int(df['year'].min())} - {int(df['year'].max())}")

    if 'citationCount' in df.columns:
        logger.info(f"\n📈 Citations:")
        logger.info(f"  Mean: {df['citationCount'].mean():.2f}")
        logger.info(f"  Median: {df['citationCount'].median():.0f}")
        logger.info(f"  Max: {df['citationCount'].max():,.0f}")

    abstracts = sum(1 for p in valid_papers if p.get('abstract'))
    h_index = sum(1 for p in valid_papers if any(a.get('hIndex') for a in p.get('authors', [])))

    logger.info(f"\n✅ Papers with abstracts: {abstracts:,} ({abstracts/len(valid_papers)*100:.1f}%)")
    logger.info(f"✅ Papers with author h-index: {h_index:,} ({h_index/len(valid_papers)*100:.1f}%)")

    logger.info("\n" + "="*60)
    logger.info("✅ Cleaning Complete!")
    logger.info("="*60)

    return len(valid_papers)


def main():
    """Clean all collected datasets."""

    logger.info("="*60)
    logger.info("=== CitaPred Data Cleaning ===")
    logger.info("="*60)

    datasets = [
        {
            'input': 'data/raw/aub_papers_all.json',
            'output': 'data/processed/aub_papers_clean.json',
            'name': 'AUB Papers (All)'
        },
        {
            'input': 'data/raw/aub_papers_complete.json',
            'output': 'data/processed/aub_papers_complete_clean.json',
            'name': 'AUB Papers (Complete)'
        },
        {
            'input': 'data/raw/complete_dataset.json',
            'output': 'data/processed/general_ml_papers_clean.json',
            'name': 'General ML Papers'
        }
    ]

    for dataset in datasets:
        input_path = Path(dataset['input'])

        if not input_path.exists():
            logger.warning(f"\n⚠️  Skipping {dataset['name']}: File not found")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Cleaning: {dataset['name']}")
        logger.info(f"{'='*60}")

        try:
            count = clean_dataset(dataset['input'], dataset['output'])
            logger.info(f"\n✅ {dataset['name']}: {count:,} clean papers")
        except Exception as e:
            logger.error(f"❌ Error cleaning {dataset['name']}: {e}")

    logger.info("\n" + "="*60)
    logger.info("🎉 All datasets cleaned!")
    logger.info("="*60)
    logger.info("\nCleaned files saved to: data/processed/")
    logger.info("\nNext steps:")
    logger.info("  1. Review cleaned CSV files")
    logger.info("  2. Train models on cleaned data")
    logger.info("  3. Compare AUB vs General ML results")


if __name__ == "__main__":
    main()
