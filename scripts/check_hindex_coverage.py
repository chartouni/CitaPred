"""
Diagnose h-index data availability in collected papers.
"""

import json
import sys
from pathlib import Path

def check_hindex_data(json_file):
    """Check if h-index data exists in a dataset."""

    print("="*60)
    print(f"Checking: {json_file}")
    print("="*60)

    with open(json_file, 'r') as f:
        papers = json.load(f)

    print(f"\nTotal papers: {len(papers):,}")

    # Count papers with h-index data
    papers_with_hindex = 0
    total_authors = 0
    authors_with_hindex = 0

    for paper in papers:
        authors = paper.get('authors', [])
        total_authors += len(authors)

        paper_has_hindex = False
        for author in authors:
            if author.get('hIndex') is not None:
                authors_with_hindex += 1
                paper_has_hindex = True

        if paper_has_hindex:
            papers_with_hindex += 1

    print(f"\nPapers with at least one h-index: {papers_with_hindex:,} ({papers_with_hindex/len(papers)*100:.1f}%)")
    print(f"Total authors: {total_authors:,}")
    print(f"Authors with h-index: {authors_with_hindex:,} ({authors_with_hindex/total_authors*100:.1f}%)")

    # Show sample
    print("\nSample of first 10 papers:")
    print("-"*60)

    for i, paper in enumerate(papers[:10], 1):
        title = paper.get('title', 'N/A')[:60]
        authors = paper.get('authors', [])

        print(f"\n{i}. {title}")

        if authors:
            author = authors[0]
            print(f"   First author: {author.get('name', 'N/A')}")
            print(f"   H-index: {author.get('hIndex', 'NONE')}")
            print(f"   Citations: {author.get('citationCount', 'NONE')}")

    print("\n" + "="*60)

    if papers_with_hindex == 0:
        print("❌ PROBLEM: No h-index data found!")
        print("   Possible causes:")
        print("   1. Data was collected without h-index fields")
        print("   2. API didn't return h-index data")
        print("   Solution: Re-collect data with proper fields")
    elif papers_with_hindex < len(papers) * 0.5:
        print("⚠️  WARNING: Low h-index coverage")
        print(f"   Only {papers_with_hindex/len(papers)*100:.1f}% of papers have h-index")
        print("   This will hurt model performance (h-index is 2nd most important feature)")
    else:
        print("✅ Good h-index coverage!")


def main():
    files_to_check = [
        'data/raw/aub_papers_all.json',
        'data/raw/aub_papers_complete.json',
        'data/processed/aub_papers_clean.json',
        'data/raw/complete_dataset.json',
    ]

    for file_path in files_to_check:
        if Path(file_path).exists():
            check_hindex_data(file_path)
            print("\n\n")
        else:
            print(f"⚠️  {file_path} not found, skipping\n")


if __name__ == "__main__":
    main()
