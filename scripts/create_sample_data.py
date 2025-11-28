"""
Create a sample dataset for testing the citation prediction model.

This generates realistic synthetic data with similar distribution to real papers.
"""

import json
import numpy as np
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sample_papers(n_samples=500):
    """Generate sample papers with realistic citation distributions."""

    papers = []

    # Sample venues
    venues = [
        "NeurIPS", "ICML", "CVPR", "ICCV", "ACL", "EMNLP", "ICLR",
        "AAAI", "IJCAI", "KDD", "WWW", "SIGIR", "ECCV", "NAACL"
    ]

    # Sample authors
    author_pool = [
        "J. Smith", "A. Johnson", "M. Williams", "K. Brown", "R. Davis",
        "S. Wilson", "L. Martinez", "C. Anderson", "D. Taylor", "E. Thomas"
    ]

    # Topics for title generation
    topics = [
        "neural networks", "deep learning", "machine learning", "computer vision",
        "natural language processing", "reinforcement learning", "transformers",
        "attention mechanisms", "convolutional networks", "recurrent networks",
        "graph neural networks", "meta-learning", "transfer learning", "few-shot learning"
    ]

    for i in range(n_samples):
        # Generate citation count with realistic distribution
        # Most papers have low citations, some have moderate, few have very high
        if np.random.random() < 0.6:  # 60% low citations
            citations = int(np.random.lognormal(2.0, 1.0))
        elif np.random.random() < 0.85:  # 25% moderate citations
            citations = int(np.random.lognormal(3.5, 0.8))
        else:  # 15% high citations
            citations = int(np.random.lognormal(5.0, 1.0))

        citations = max(0, citations)  # Ensure non-negative

        # Generate year (2015-2020)
        year = random.randint(2015, 2020)

        # Generate reference count (correlated with citations)
        base_refs = int(np.random.lognormal(3.0, 0.5))
        reference_count = max(5, min(100, base_refs))

        # Generate number of authors (1-10, most papers have 2-5)
        n_authors = min(10, max(1, int(np.random.lognormal(1.2, 0.5))))
        authors = [{"name": random.choice(author_pool)} for _ in range(n_authors)]

        # Generate title
        topic = random.choice(topics)
        title_templates = [
            f"A study on {topic}",
            f"Deep {topic} for improved performance",
            f"Learning {topic} with neural networks",
            f"Efficient {topic} using attention",
            f"Novel approach to {topic}",
            f"{topic.capitalize()}: A comprehensive study"
        ]
        title = random.choice(title_templates)

        # Generate abstract
        abstract = (
            f"This paper presents a novel approach to {topic}. "
            f"We propose a new method that improves upon existing techniques. "
            f"Our experiments demonstrate significant improvements on benchmark datasets. "
            f"The proposed method achieves state-of-the-art results in various tasks."
        )

        paper = {
            "paperId": f"sample_{i:04d}",
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "venue": random.choice(venues),
            "year": year,
            "citationCount": citations,
            "referenceCount": reference_count
        }

        papers.append(paper)

    return papers


def main():
    """Create sample dataset."""
    print("Creating sample dataset...")

    # Generate papers
    papers = generate_sample_papers(n_samples=500)

    # Create data directory
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    output_file = data_dir / "complete_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2)

    print(f"Created sample dataset with {len(papers)} papers")
    print(f"Saved to: {output_file}")

    # Print statistics
    citations = [p['citationCount'] for p in papers]
    print(f"\nCitation Statistics:")
    print(f"  Mean: {np.mean(citations):.2f}")
    print(f"  Median: {np.median(citations):.2f}")
    print(f"  Std: {np.std(citations):.2f}")
    print(f"  Min: {min(citations)}")
    print(f"  Max: {max(citations)}")
    print(f"  99th percentile: {np.percentile(citations, 99):.2f}")


if __name__ == "__main__":
    main()
