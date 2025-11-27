"""
Model implementations for citation prediction.
"""

from citapred.models.predictor import CitationPredictor
from citapred.models.baseline import BaselineModel

# Import neural network if available
try:
    from citapred.models.neural import NeuralNetworkModel
    __all__ = ["CitationPredictor", "BaselineModel", "NeuralNetworkModel"]
except ImportError:
    __all__ = ["CitationPredictor", "BaselineModel"]
