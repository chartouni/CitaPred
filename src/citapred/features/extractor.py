"""
Feature extraction for research papers.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features from research paper data for citation prediction.
    """

    def __init__(self, max_features: int = 1000, use_tfidf: bool = False):
        """
        Initialize the feature extractor.

        Args:
            max_features: Maximum number of text features to extract
            use_tfidf: Whether to extract TF-IDF features from titles/abstracts
        """
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        self.title_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.abstract_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.is_fitted = False

        # Venue prestige scores (based on typical citation impact)
        self.venue_prestige = {
            # Top tier - Nature, Science family
            'Nature': 10.0, 'Science': 10.0, 'Cell': 9.5,
            'Nature Medicine': 9.0, 'Nature Genetics': 9.0, 'Nature Biotechnology': 9.0,

            # Top ML/AI conferences
            'NeurIPS': 9.0, 'NIPS': 9.0, 'ICML': 9.0, 'ICLR': 8.5,
            'CVPR': 8.5, 'ICCV': 8.5, 'ECCV': 8.0,
            'ACL': 8.0, 'EMNLP': 7.5, 'NAACL': 7.0,

            # Top systems/theory conferences
            'OSDI': 8.5, 'SOSP': 8.5, 'SIGCOMM': 8.0, 'NSDI': 8.0,
            'FOCS': 8.5, 'STOC': 8.5, 'SODA': 7.5,

            # Top AI conferences
            'AAAI': 7.5, 'IJCAI': 7.5, 'KDD': 8.0, 'WWW': 7.5,

            # Good venues
            'SIGIR': 7.0, 'CIKM': 6.5, 'WSDM': 6.5,
            'ICRA': 7.0, 'IROS': 6.5, 'RSS': 7.5,

            # Popular journals
            'PLOS ONE': 5.0, 'Scientific Reports': 5.0,
            'IEEE Transactions': 6.5, 'ACM Transactions': 6.5,

            # Medical
            'The Lancet': 9.5, 'NEJM': 10.0, 'JAMA': 9.0, 'BMJ': 8.0,

            # Arxiv (preprints)
            'arXiv': 4.0, 'bioRxiv': 4.0
        }

        # Compute venue statistics on fit
        self.venue_citation_stats = {}

    def fit(self, df: pd.DataFrame):
        """
        Fit the feature extractors on the training data.

        Args:
            df: DataFrame containing paper data
        """
        if 'title' in df.columns and not df['title'].isna().all():
            self.title_vectorizer.fit(df['title'].fillna(''))

        if 'abstract' in df.columns and not df['abstract'].isna().all():
            self.abstract_vectorizer.fit(df['abstract'].fillna(''))

        # Compute venue statistics from training data
        if 'venue' in df.columns and 'citationCount' in df.columns:
            venue_stats = df.groupby('venue')['citationCount'].agg(['mean', 'median', 'std', 'count'])
            self.venue_citation_stats = venue_stats.to_dict('index')

        self.is_fitted = True
        logger.info("Feature extractors fitted successfully")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform paper data into features.

        Args:
            df: DataFrame containing paper data

        Returns:
            DataFrame with extracted features
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")

        features = pd.DataFrame()

        # Metadata features
        features = self._extract_metadata_features(df, features)

        # Venue features
        features = self._extract_venue_features(df, features)

        # Time-based features
        features = self._extract_time_features(df, features)

        # Author features
        features = self._extract_author_features(df, features)

        # Interaction features
        features = self._extract_interaction_features(df, features)

        # Text features (TF-IDF) - controlled by use_tfidf flag
        if self.use_tfidf:
            features = self._extract_text_features(df, features)

        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame containing paper data

        Returns:
            DataFrame with extracted features
        """
        self.fit(df)
        return self.transform(df)

    def _extract_metadata_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract metadata features.

        Args:
            df: Input DataFrame
            features: Features DataFrame to append to

        Returns:
            Updated features DataFrame
        """
        # Year
        if 'year' in df.columns:
            features['year'] = df['year'].fillna(0)

        # Reference count
        if 'referenceCount' in df.columns:
            features['reference_count'] = df['referenceCount'].fillna(0)

        # Title length
        if 'title' in df.columns:
            features['title_length'] = df['title'].fillna('').apply(len)
            features['title_word_count'] = df['title'].fillna('').apply(lambda x: len(x.split()))

        # Abstract length
        if 'abstract' in df.columns:
            features['abstract_length'] = df['abstract'].fillna('').apply(len)
            features['abstract_word_count'] = df['abstract'].fillna('').apply(lambda x: len(x.split()))

        # Has abstract
        if 'abstract' in df.columns:
            features['has_abstract'] = df['abstract'].fillna('').apply(lambda x: 1 if len(x) > 0 else 0)

        return features

    def _extract_author_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract author-related features.

        Args:
            df: Input DataFrame
            features: Features DataFrame to append to

        Returns:
            Updated features DataFrame
        """
        # Author count
        if 'authors' in df.columns:
            features['author_count'] = df['authors'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

            # Extract author h-index statistics (if available)
            def get_author_h_indices(authors):
                """Extract h-indices from author list."""
                if not isinstance(authors, list):
                    return []
                h_indices = []
                for author in authors:
                    if isinstance(author, dict) and 'hIndex' in author:
                        h_indices.append(author['hIndex'])
                return h_indices

            # Max, mean, and sum of author h-indices
            h_index_lists = df['authors'].apply(get_author_h_indices)
            features['max_author_hindex'] = h_index_lists.apply(
                lambda x: max(x) if len(x) > 0 else 0
            )
            features['mean_author_hindex'] = h_index_lists.apply(
                lambda x: np.mean(x) if len(x) > 0 else 0
            )
            features['sum_author_hindex'] = h_index_lists.apply(
                lambda x: sum(x) if len(x) > 0 else 0
            )

            # Extract author citation counts (if available)
            def get_author_citations(authors):
                """Extract citation counts from author list."""
                if not isinstance(authors, list):
                    return []
                citations = []
                for author in authors:
                    if isinstance(author, dict) and 'citationCount' in author:
                        citations.append(author['citationCount'])
                return citations

            citation_lists = df['authors'].apply(get_author_citations)
            features['max_author_citations'] = citation_lists.apply(
                lambda x: max(x) if len(x) > 0 else 0
            )
            features['mean_author_citations'] = citation_lists.apply(
                lambda x: np.mean(x) if len(x) > 0 else 0
            )

        elif 'author_count' in df.columns:
            features['author_count'] = df['author_count'].fillna(0)

        return features

    def _extract_venue_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract venue-related features.

        Args:
            df: Input DataFrame
            features: Features DataFrame to append to

        Returns:
            Updated features DataFrame
        """
        if 'venue' not in df.columns:
            return features

        # Venue prestige score (manual mapping)
        def get_venue_prestige(venue_name):
            """Get prestige score for a venue."""
            if pd.isna(venue_name) or venue_name == '':
                return 3.0  # Default for unknown venues

            venue_name = str(venue_name).strip()

            # Exact match
            if venue_name in self.venue_prestige:
                return self.venue_prestige[venue_name]

            # Partial match (case insensitive)
            venue_lower = venue_name.lower()
            for known_venue, score in self.venue_prestige.items():
                if known_venue.lower() in venue_lower or venue_lower in known_venue.lower():
                    return score

            # Default for unknown venues
            return 3.0

        features['venue_prestige'] = df['venue'].apply(get_venue_prestige)

        # Venue citation statistics (learned from training data)
        def get_venue_stat(venue_name, stat_name, default_value):
            """Get venue statistic from training data."""
            if pd.isna(venue_name) or venue_name not in self.venue_citation_stats:
                return default_value
            stats = self.venue_citation_stats[venue_name]
            return stats.get(stat_name, default_value)

        features['venue_mean_citations'] = df['venue'].apply(
            lambda v: get_venue_stat(v, 'mean', features['venue_prestige'].median())
        )
        features['venue_median_citations'] = df['venue'].apply(
            lambda v: get_venue_stat(v, 'median', 0)
        )
        features['venue_paper_count'] = df['venue'].apply(
            lambda v: get_venue_stat(v, 'count', 1)
        )

        # Venue is top tier (prestige >= 8)
        features['is_top_venue'] = (features['venue_prestige'] >= 8.0).astype(int)

        return features

    def _extract_time_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features.

        Args:
            df: Input DataFrame
            features: Features DataFrame to append to

        Returns:
            Updated features DataFrame
        """
        if 'year' not in df.columns:
            return features

        # Compute years since publication (assuming measurement year is 2023)
        current_year = 2023
        features['years_since_pub'] = df['year'].apply(
            lambda y: max(0, current_year - y) if pd.notna(y) else 0
        )

        # Publication age categories
        features['is_recent'] = (features['years_since_pub'] <= 2).astype(int)
        features['is_classic'] = (features['years_since_pub'] >= 10).astype(int)

        # Year squared (non-linear time effect)
        features['year_squared'] = features['year'] ** 2

        return features

    def _extract_interaction_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract interaction features between different dimensions.

        Args:
            df: Input DataFrame
            features: Features DataFrame to append to

        Returns:
            Updated features DataFrame
        """
        # Venue × Time interactions
        if 'venue_prestige' in features and 'year' in features:
            features['venue_prestige_x_year'] = features['venue_prestige'] * features['year']
            features['venue_prestige_x_recency'] = features['venue_prestige'] * (1 / (features['years_since_pub'] + 1))

        # Author × Reference interactions
        if 'author_count' in features and 'reference_count' in features:
            features['authors_x_refs'] = features['author_count'] * features['reference_count']
            features['refs_per_author'] = features['reference_count'] / (features['author_count'] + 1)

        # Venue × References
        if 'venue_prestige' in features and 'reference_count' in features:
            features['venue_x_refs'] = features['venue_prestige'] * features['reference_count']

        # Title length × Venue (shorter titles in top venues might indicate focused work)
        if 'venue_prestige' in features and 'title_length' in features:
            features['title_len_x_venue'] = features['title_length'] * features['venue_prestige']

        return features

    def _extract_text_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract text-based features using TF-IDF.

        Args:
            df: Input DataFrame
            features: Features DataFrame to append to

        Returns:
            Updated features DataFrame
        """
        # Title TF-IDF
        if 'title' in df.columns:
            title_tfidf = self.title_vectorizer.transform(df['title'].fillna(''))
            title_features = pd.DataFrame(
                title_tfidf.toarray(),
                columns=[f'title_tfidf_{i}' for i in range(title_tfidf.shape[1])],
                index=df.index
            )
            features = pd.concat([features, title_features], axis=1)

        # Abstract TF-IDF
        if 'abstract' in df.columns:
            abstract_tfidf = self.abstract_vectorizer.transform(df['abstract'].fillna(''))
            abstract_features = pd.DataFrame(
                abstract_tfidf.toarray(),
                columns=[f'abstract_tfidf_{i}' for i in range(abstract_tfidf.shape[1])],
                index=df.index
            )
            features = pd.concat([features, abstract_features], axis=1)

        return features
