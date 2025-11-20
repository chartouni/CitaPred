"""
Data preprocessing and cleaning utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses and cleans research paper data.
    """

    def __init__(self):
        """Initialize the preprocessor."""
        pass

    def clean_paper_data(self, papers: List[Dict]) -> pd.DataFrame:
        """
        Clean and normalize paper data.

        Args:
            papers: List of paper dictionaries

        Returns:
            Cleaned pandas DataFrame
        """
        df = pd.DataFrame(papers)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Clean text fields
        if 'title' in df.columns:
            df['title'] = df['title'].apply(self._clean_text)
        if 'abstract' in df.columns:
            df['abstract'] = df['abstract'].apply(self._clean_text)

        # Extract author count
        if 'authors' in df.columns:
            df['author_count'] = df['authors'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled missing values
        """
        # Fill missing citation counts with 0
        if 'citationCount' in df.columns:
            df['citationCount'] = df['citationCount'].fillna(0)

        # Fill missing reference counts with 0
        if 'referenceCount' in df.columns:
            df['referenceCount'] = df['referenceCount'].fillna(0)

        # Fill missing text with empty string
        text_columns = ['title', 'abstract', 'venue']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        return df

    def _clean_text(self, text: Optional[str]) -> str:
        """
        Clean text data.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Convert to lowercase (optional, depends on your model)
        # text = text.lower()

        return text

    def split_by_year(self, df: pd.DataFrame, train_end_year: int,
                      val_end_year: int) -> tuple:
        """
        Split data by publication year for temporal validation.

        Args:
            df: Input DataFrame
            train_end_year: Last year for training data
            val_end_year: Last year for validation data

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if 'year' not in df.columns:
            raise ValueError("DataFrame must contain 'year' column")

        train_df = df[df['year'] <= train_end_year]
        val_df = df[(df['year'] > train_end_year) & (df['year'] <= val_end_year)]
        test_df = df[df['year'] > val_end_year]

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df
