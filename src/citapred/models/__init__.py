"""
Model implementations for citation prediction.
"""

from citapred.models.predictor import CitationPredictor
from citapred.models.baseline import BaselineModel

__all__ = ["CitationPredictor", "BaselineModel"]
