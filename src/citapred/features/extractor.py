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

    def __init__(self, max_features: int = 1000):
        """
        Initialize the feature extractor.

        Args:
            max_features: Maximum number of text features to extract
        """
        self.max_features = max_features
        self.title_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.abstract_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.is_fitted = False

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

        # Author features
        features = self._extract_author_features(df, features)

        # Text features (optional - can be memory intensive)
        # Uncomment if you want to use text features
        # features = self._extract_text_features(df, features)

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
