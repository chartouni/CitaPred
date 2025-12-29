"""
Exploratory Data Analysis for Official AUB Data.

This script analyzes the official dataset received from AUB (likely Scopus export)
and outputs all feature names and comprehensive statistics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citapred.utils.logger import setup_logger

logger = setup_logger()


def load_aub_data(file_path: str) -> pd.DataFrame:
    """Load AUB data from various file formats."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Reading file: {file_path}")
    logger.info(f"File size: {file_path.stat().st_size / 1024:.2f} KB")

    # Try different formats
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.csv':
        # Try different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except:
                df = pd.read_csv(file_path, encoding='cp1252')
    elif file_path.suffix.lower() in ['.tsv', '.txt']:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    else:
        # Try CSV as default
        df = pd.read_csv(file_path)

    logger.info(f"✅ Successfully loaded {len(df):,} rows\n")
    return df


def analyze_columns(df: pd.DataFrame):
    """Analyze all columns in the dataset."""

    logger.info("="*70)
    logger.info("📋 COLUMN NAMES AND DATA TYPES")
    logger.info("="*70)

    logger.info(f"\nTotal columns: {len(df.columns)}\n")

    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        unique_count = df[col].nunique()

        logger.info(f"{i:2d}. {col}")
        logger.info(f"    Type: {dtype}")
        logger.info(f"    Non-null: {non_null:,} ({100-null_pct:.1f}%)")
        logger.info(f"    Unique values: {unique_count:,}")

        # Show sample values for better understanding
        sample_values = df[col].dropna().head(3).tolist()
        if sample_values:
            logger.info(f"    Sample: {sample_values}")
        logger.info("")


def analyze_basic_stats(df: pd.DataFrame):
    """Show basic dataset statistics."""

    logger.info("="*70)
    logger.info("📊 BASIC DATASET STATISTICS")
    logger.info("="*70)

    logger.info(f"\n📄 Total papers: {len(df):,}")
    logger.info(f"📋 Total columns: {len(df.columns)}")
    logger.info(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Missing values summary
    logger.info("\n❓ Missing Values:")
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percent': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)

    if len(missing_df) > 0:
        for _, row in missing_df.iterrows():
            logger.info(f"  {row['Column']}: {row['Missing']:,} ({row['Percent']:.1f}%)")
    else:
        logger.info("  ✅ No missing values!")

    # Duplicates
    duplicates = df.duplicated().sum()
    logger.info(f"\n🔄 Duplicate rows: {duplicates:,}")

    if duplicates > 0:
        logger.info(f"   ({duplicates/len(df)*100:.1f}% of dataset)")


def analyze_year_distribution(df: pd.DataFrame):
    """Analyze publication year distribution."""

    # Try to find year column
    year_col = None
    for col in df.columns:
        if 'year' in col.lower():
            year_col = col
            break

    if not year_col:
        logger.info("\n⚠️  No year column found")
        return

    logger.info("="*70)
    logger.info("📅 YEAR DISTRIBUTION")
    logger.info("="*70)

    years = df[year_col].dropna()

    if len(years) == 0:
        logger.info("\n⚠️  No year data available")
        return

    logger.info(f"\nYear range: {int(years.min())} - {int(years.max())}")
    logger.info(f"Mean year: {years.mean():.1f}")
    logger.info(f"Median year: {int(years.median())}")

    # Papers by decade
    logger.info("\n📊 Papers by Decade:")
    decades = (years // 10) * 10
    decade_counts = decades.value_counts().sort_index(ascending=False)

    for decade, count in decade_counts.head(10).items():
        logger.info(f"  {int(decade)}s: {count:,} papers")

    # Recent years (last 10 years)
    recent_years = years[years >= years.max() - 10]
    logger.info(f"\n📈 Last 10 years: {len(recent_years):,} papers ({len(recent_years)/len(years)*100:.1f}%)")


def analyze_authors(df: pd.DataFrame):
    """Analyze author information."""

    # Try to find author column
    author_col = None
    for col in df.columns:
        if 'author' in col.lower():
            author_col = col
            break

    if not author_col:
        logger.info("\n⚠️  No author column found")
        return

    logger.info("="*70)
    logger.info("👥 AUTHOR ANALYSIS")
    logger.info("="*70)

    authors = df[author_col].dropna()

    logger.info(f"\nPapers with author data: {len(authors):,} ({len(authors)/len(df)*100:.1f}%)")

    # Try to count authors per paper
    # Common separators: comma, semicolon, "and"
    author_counts = []
    for author_str in authors.head(100):  # Sample first 100
        if isinstance(author_str, str):
            # Count separators
            count = max(
                author_str.count(','),
                author_str.count(';'),
                author_str.count(' and ')
            ) + 1
            author_counts.append(count)

    if author_counts:
        logger.info(f"\nEstimated authors per paper (from sample):")
        logger.info(f"  Mean: {np.mean(author_counts):.1f}")
        logger.info(f"  Median: {np.median(author_counts):.0f}")
        logger.info(f"  Max: {max(author_counts)}")


def analyze_venues(df: pd.DataFrame):
    """Analyze publication venues."""

    # Try to find venue/source column
    venue_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['venue', 'source', 'journal', 'conference']):
            venue_col = col
            break

    if not venue_col:
        logger.info("\n⚠️  No venue column found")
        return

    logger.info("="*70)
    logger.info("🏛️  VENUE ANALYSIS")
    logger.info("="*70)

    venues = df[venue_col].dropna()

    logger.info(f"\nPapers with venue data: {len(venues):,} ({len(venues)/len(df)*100:.1f}%)")
    logger.info(f"Unique venues: {venues.nunique():,}")

    # Top venues
    logger.info("\n🏆 Top 20 Publication Venues:")
    top_venues = venues.value_counts().head(20)

    for venue, count in top_venues.items():
        venue_str = str(venue)[:60]
        logger.info(f"  {count:4d} - {venue_str}")


def analyze_citation_fields(df: pd.DataFrame):
    """Check for citation-related fields."""

    logger.info("="*70)
    logger.info("📈 CITATION-RELATED FIELDS")
    logger.info("="*70)

    citation_keywords = ['citation', 'cited', 'impact', 'score', 'snip', 'sjr', 'if', 'h-index', 'quartile']

    found_fields = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in citation_keywords):
            found_fields.append(col)

    if found_fields:
        logger.info(f"\n✅ Found {len(found_fields)} citation-related fields:")
        for field in found_fields:
            non_null = df[field].notna().sum()
            logger.info(f"\n  • {field}")
            logger.info(f"    Non-null: {non_null:,} ({non_null/len(df)*100:.1f}%)")

            # Try to show numeric statistics
            if pd.api.types.is_numeric_dtype(df[field]):
                values = df[field].dropna()
                if len(values) > 0:
                    logger.info(f"    Mean: {values.mean():.2f}")
                    logger.info(f"    Median: {values.median():.2f}")
                    logger.info(f"    Range: {values.min():.2f} - {values.max():.2f}")
    else:
        logger.info("\n⚠️  No citation-related fields found")
        logger.info("   → Will need to enrich with Semantic Scholar API")


def show_sample_data(df: pd.DataFrame):
    """Show sample rows from the dataset."""

    logger.info("="*70)
    logger.info("📄 SAMPLE DATA (First 5 Rows)")
    logger.info("="*70)

    # Show key columns only to avoid clutter
    key_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['title', 'author', 'year', 'venue', 'source', 'citation']):
            key_cols.append(col)

    if not key_cols:
        key_cols = df.columns[:5].tolist()

    logger.info(f"\nShowing columns: {', '.join(key_cols)}\n")

    sample = df[key_cols].head(5)
    for idx, row in sample.iterrows():
        logger.info(f"Row {idx + 1}:")
        for col in key_cols:
            value = str(row[col])[:80]
            logger.info(f"  {col}: {value}")
        logger.info("")


def generate_recommendations(df: pd.DataFrame):
    """Generate recommendations for next steps."""

    logger.info("="*70)
    logger.info("💡 RECOMMENDATIONS")
    logger.info("="*70)

    # Check for citation data
    has_citations = any('citation' in col.lower() or 'cited' in col.lower()
                       for col in df.columns)

    # Check for year
    has_year = any('year' in col.lower() for col in df.columns)

    # Check for title
    has_title = any('title' in col.lower() for col in df.columns)

    logger.info("\n📋 Data Completeness:")
    logger.info(f"  {'✅' if has_title else '❌'} Title field")
    logger.info(f"  {'✅' if has_year else '❌'} Year field")
    logger.info(f"  {'✅' if has_citations else '❌'} Citation data")

    logger.info("\n🔄 Next Steps:")

    if not has_citations:
        logger.info("\n1. ENRICH DATA WITH SEMANTIC SCHOLAR:")
        logger.info("   - Match papers by title/DOI to get citation counts")
        logger.info("   - Fetch author h-indices")
        logger.info("   - Get abstract text")
        logger.info("   - Script: scripts/enrich_aub_data.py (need to create)")

    logger.info("\n2. DATA CLEANING:")
    logger.info("   - Remove duplicates")
    logger.info("   - Filter invalid years")
    logger.info("   - Handle missing values")
    logger.info("   - Standardize author names")

    logger.info("\n3. FEATURE ENGINEERING:")
    logger.info("   - Extract text features from titles/abstracts")
    logger.info("   - Calculate author metrics")
    logger.info("   - Add venue prestige scores")
    logger.info("   - Create temporal features")

    logger.info("\n4. MODEL TRAINING:")
    logger.info("   - Train classification model (highly cited vs not)")
    logger.info("   - Train regression model (citation count prediction)")
    logger.info("   - Compare with general ML model")

    # Estimate data quality
    completeness = df.notna().sum().sum() / (len(df) * len(df.columns)) * 100
    logger.info(f"\n📊 Overall Data Completeness: {completeness:.1f}%")

    if completeness > 80:
        logger.info("   ✅ Good data quality!")
    elif completeness > 60:
        logger.info("   ⚠️  Moderate data quality - some enrichment needed")
    else:
        logger.info("   ❌ Low data quality - significant enrichment required")


def main():
    """Run comprehensive EDA on official AUB data."""

    logger.info("="*70)
    logger.info("=== Exploratory Data Analysis: Official AUB Data ===")
    logger.info("="*70)

    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Try common file names/locations
        possible_paths = [
            'data/raw/aub_official.xlsx',
            'data/raw/aub_official.csv',
            'data/raw/aub_data.xlsx',
            'data/raw/aub_data.csv',
            'data/raw/scopus_export.xlsx',
            'data/raw/scopus_export.csv',
        ]

        file_path = None
        for path in possible_paths:
            if Path(path).exists():
                file_path = path
                break

        if not file_path:
            logger.error("\n❌ No data file found!")
            logger.info("\nUsage: python scripts/eda_aub_official.py <path_to_file>")
            logger.info("\nExpected file locations:")
            for path in possible_paths:
                logger.info(f"  - {path}")
            return

    try:
        # Load data
        df = load_aub_data(file_path)

        # Run analyses
        analyze_columns(df)
        analyze_basic_stats(df)
        analyze_year_distribution(df)
        analyze_authors(df)
        analyze_venues(df)
        analyze_citation_fields(df)
        show_sample_data(df)
        generate_recommendations(df)

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("✅ EDA COMPLETE")
        logger.info("="*70)
        logger.info(f"\n📊 Dataset Summary:")
        logger.info(f"   Papers: {len(df):,}")
        logger.info(f"   Features: {len(df.columns)}")
        logger.info(f"   File: {file_path}")
        logger.info("\n" + "="*70)

    except Exception as e:
        logger.error(f"\n❌ Error during EDA: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
