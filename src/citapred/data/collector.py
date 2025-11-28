"""
Data collection from various sources (Semantic Scholar, ArXiv, etc.).
"""

import requests
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects research paper data from various APIs and sources.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data collector.

        Args:
            api_key: API key for services that require authentication
        """
        self.api_key = api_key
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"

    def fetch_paper_by_id(self, paper_id: str, source: str = "semantic_scholar") -> Optional[Dict]:
        """
        Fetch paper metadata by paper ID.

        Args:
            paper_id: The paper identifier
            source: The data source (semantic_scholar, arxiv, etc.)

        Returns:
            Dictionary containing paper metadata or None if not found
        """
        if source == "semantic_scholar":
            return self._fetch_from_semantic_scholar(paper_id)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def _fetch_from_semantic_scholar(self, paper_id: str) -> Optional[Dict]:
        """
        Fetch paper data from Semantic Scholar API.

        Args:
            paper_id: Semantic Scholar paper ID or DOI

        Returns:
            Dictionary containing paper metadata
        """
        url = f"{self.semantic_scholar_base_url}/paper/{paper_id}"
        params = {
            "fields": "title,abstract,authors,venue,year,citationCount,referenceCount,influentialCitationCount"
        }

        # Add API key to headers if available
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None

    def search_papers(self, query: str, limit: int = 100, offset: int = 0, source: str = "semantic_scholar") -> List[Dict]:
        """
        Search for papers by query.

        Args:
            query: Search query string
            limit: Maximum number of results per request
            offset: Starting position for pagination
            source: The data source

        Returns:
            List of paper metadata dictionaries
        """
        if source == "semantic_scholar":
            return self._search_semantic_scholar(query, limit, offset)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def _search_semantic_scholar(self, query: str, limit: int, offset: int = 0) -> List[Dict]:
        """
        Search papers on Semantic Scholar.

        Args:
            query: Search query
            limit: Maximum results per request
            offset: Starting position for pagination

        Returns:
            List of papers
        """
        url = f"{self.semantic_scholar_base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": "title,abstract,authors.name,authors.hIndex,authors.citationCount,venue,year,citationCount,referenceCount"
        }

        # Add API key to headers if available
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching papers: {e}")
            return []
