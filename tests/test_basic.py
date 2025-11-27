"""
Basic tests for CitaPred modules.
"""

import pytest
import pandas as pd
import numpy as np
from citapred.data.preprocessor import DataPreprocessor
from citapred.features.extractor import FeatureExtractor
from citapred.models.baseline import BaselineModel

# Check if PyTorch is available for neural network tests
try:
    from citapred.models.neural import NeuralNetworkModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_clean_paper_data(self):
        """Test basic data cleaning."""
        papers = [
            {
                'title': 'Test Paper 1',
                'abstract': 'This is a test abstract.',
                'year': 2020,
                'citationCount': 10,
                'referenceCount': 5,
                'authors': [{'name': 'Author 1'}, {'name': 'Author 2'}]
            },
            {
                'title': 'Test Paper 2',
                'abstract': None,
                'year': 2021,
                'citationCount': None,
                'referenceCount': 3,
                'authors': [{'name': 'Author 3'}]
            }
        ]

        preprocessor = DataPreprocessor()
        df = preprocessor.clean_paper_data(papers)

        assert len(df) == 2
        assert 'author_count' in df.columns
        assert df['author_count'].iloc[0] == 2
        assert df['citationCount'].iloc[1] == 0  # Should be filled with 0


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_feature_extraction(self):
        """Test feature extraction."""
        df = pd.DataFrame({
            'title': ['Paper 1', 'Paper 2', 'Paper 3'],
            'abstract': ['Abstract 1', 'Abstract 2', 'Abstract 3'],
            'year': [2020, 2021, 2022],
            'referenceCount': [5, 10, 15],
            'authors': [[1, 2], [1], [1, 2, 3]]
        })

        extractor = FeatureExtractor(max_features=10)
        features = extractor.fit_transform(df)

        assert len(features) == 3
        assert 'year' in features.columns
        assert 'reference_count' in features.columns
        assert 'title_length' in features.columns
        assert 'author_count' in features.columns


class TestBaselineModel:
    """Tests for BaselineModel class."""

    def test_linear_model(self):
        """Test linear regression model."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100) * 100

        model = BaselineModel(model_type='linear')
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert predictions.shape == y.shape

    def test_mean_model(self):
        """Test mean baseline model."""
        X = np.random.rand(100, 5)
        y = np.array([10] * 50 + [20] * 50)

        model = BaselineModel(model_type='mean')
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert np.allclose(predictions, 15.0)  # Mean of 10 and 20


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestNeuralNetworkModel:
    """Tests for NeuralNetworkModel class."""

    def test_neural_net_initialization(self):
        """Test neural network initialization."""
        model = NeuralNetworkModel(
            hidden_sizes=[32, 16],
            epochs=5,
            batch_size=16
        )
        assert model.hidden_sizes == [32, 16]
        assert model.epochs == 5
        assert model.batch_size == 16

    def test_neural_net_training(self):
        """Test neural network training and prediction."""
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.rand(100) * 100

        model = NeuralNetworkModel(
            hidden_sizes=[32, 16],
            epochs=10,
            batch_size=16,
            early_stopping_patience=5
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert predictions.shape == (100,)
        assert not np.isnan(predictions).any()

    def test_neural_net_score(self):
        """Test neural network R² score calculation."""
        np.random.seed(42)
        X = np.random.rand(50, 8)
        y = np.random.rand(50) * 50

        model = NeuralNetworkModel(
            hidden_sizes=[16, 8],
            epochs=10,
            batch_size=8
        )
        model.fit(X, y)
        score = model.score(X, y)

        # Score should be a valid number
        assert isinstance(score, (float, np.float32, np.float64))
        assert not np.isnan(score)

    def test_neural_net_via_baseline(self):
        """Test neural network through BaselineModel interface."""
        np.random.seed(42)
        X = np.random.rand(80, 6)
        y = np.random.rand(80) * 100

        model = BaselineModel(
            model_type='neural_net',
            hidden_sizes=[24, 12],
            epochs=10,
            batch_size=16
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 80
        assert not np.isnan(predictions).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
