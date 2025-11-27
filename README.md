# CitaPred: Research Paper Citation Predictor

A machine learning system for predicting the citation count of research papers based on paper metadata, author information, and content features.

## Overview

CitaPred analyzes research papers and predicts their future citation impact using various features including:
- Paper metadata (title, abstract, venue, publication year)
- Author features (reputation, h-index, collaboration network)
- Content features (keywords, topics, text embeddings)
- Venue features (journal impact factor, conference rankings)

## Project Structure

```
CitaPred/
├── src/citapred/          # Source code
│   ├── data/              # Data collection and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   ├── evaluation/        # Evaluation metrics and tools
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed/cleaned data
│   └── external/          # External datasets
├── models/                # Saved model files
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── config/                # Configuration files
└── requirements.txt       # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd CitaPred

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from citapred.models import CitationPredictor

# Initialize predictor
predictor = CitationPredictor()

# Train model
predictor.train(training_data)

# Make predictions
predictions = predictor.predict(paper_metadata)
```

## Data Sources

Potential data sources for research papers:
- **Semantic Scholar API**: Free API with citation data
- **ArXiv**: Preprints in physics, math, CS, etc.
- **PubMed**: Biomedical literature
- **Microsoft Academic Graph (MAG)**: Large-scale academic graph (archived)
- **OpenCitations**: Open citation data

## Features

The system extracts and engineers features in several categories:

### Paper Features
- Title and abstract text embeddings
- Number of references
- Paper length
- Publication year

### Author Features
- Total number of authors
- Author h-index (if available)
- Author previous citation counts
- Author collaboration network metrics

### Venue Features
- Journal impact factor
- Conference ranking
- Venue acceptance rate

### Temporal Features
- Publication month/season
- Years since author's first publication

## Models

Supported model architectures:
- **Linear Regression** (baseline) - Simple linear model
- **Random Forest** - Ensemble tree-based model
- **Neural Networks** - Feedforward neural network with PyTorch ✅ **IMPLEMENTED**
- Gradient Boosting (XGBoost, LightGBM) - *Coming soon*
- Graph Neural Networks (for citation network modeling) - *Coming soon*

## Evaluation

The system uses:
- **Metrics**: MAE, RMSE, R², Pearson correlation
- **Validation**: K-fold cross-validation, temporal validation
- **Benchmarks**: Comparison against baseline models

## Usage Examples

### Neural Network Example

```python
from citapred.models import CitationPredictor

# Configure neural network
config = {
    'hidden_sizes': [128, 64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 16
}

# Train neural network predictor
predictor = CitationPredictor(model_type="neural_net", config=config)
predictor.train(training_data)
predictions = predictor.predict(test_data)
```

**Note:** Neural networks require PyTorch: `pip install torch`

### Additional Examples

See the `examples/` directory for complete examples:
- `basic_usage.py`: Basic citation prediction pipeline
- `neural_network_usage.py`: Neural network training and comparison

See the `notebooks/` directory for detailed examples:
- `01_data_exploration.ipynb`: Data analysis and visualization
- `02_feature_engineering.ipynb`: Feature extraction examples
- `03_model_training.ipynb`: Model training and evaluation

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## Citation

If you use this project in your research, please cite:

```bibtex
@software{citapred,
  title={CitaPred: Research Paper Citation Predictor},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/CitaPred}
}
```

## Roadmap

- [ ] Implement data collection pipelines
- [ ] Build feature extraction modules
- [ ] Train baseline models
- [ ] Implement deep learning models
- [ ] Add citation network features
- [ ] Create web API
- [ ] Deploy as web service
