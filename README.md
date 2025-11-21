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

### Web Interface (Recommended for beginners)

Launch the interactive web interface:

```bash
# Option 1: Streamlit (Full-featured)
streamlit run app.py

# Option 2: Gradio (Simple demo)
python app_gradio.py

# Option 3: REST API
python api.py
```

Or use the launcher script:
```bash
# Linux/Mac
./run_ui.sh

# Windows
run_ui.bat
```

### Python API

```python
from citapred.models import CitationPredictor

# Initialize predictor
predictor = CitationPredictor()

# Train model
predictor.train(training_data)

# Make predictions
predictions = predictor.predict(paper_metadata)
```

### REST API

```bash
# Make a prediction via HTTP
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"title": "My Paper", "abstract": "...", "year": 2024}'
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
- Linear Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks
- Graph Neural Networks (for citation network modeling)

## Evaluation

The system uses:
- **Metrics**: MAE, RMSE, R², Pearson correlation
- **Validation**: K-fold cross-validation, temporal validation
- **Benchmarks**: Comparison against baseline models

## Web Interfaces

CitaPred provides three different web interfaces:

### 1. Streamlit Web App (`app.py`)
Full-featured web application with:
- 🔮 Single paper prediction with detailed analysis
- 📊 Model performance dashboard
- 🔍 Paper search from Semantic Scholar
- 📈 Interactive model training
- 📄 Batch predictions via CSV upload
- 📉 Interactive visualizations

**Launch:** `streamlit run app.py`
**URL:** http://localhost:8501

### 2. Gradio Interface (`app_gradio.py`)
Simple, clean interface for quick demos:
- 🎯 Quick predictions
- 🔍 Paper search
- 📈 Model training
- 📱 Mobile-friendly

**Launch:** `python app_gradio.py`
**URL:** http://localhost:7860

### 3. FastAPI REST API (`api.py`)
RESTful API for programmatic access:
- 🔌 RESTful endpoints
- 📚 Auto-generated documentation
- 🚀 High performance
- 🔐 CORS enabled

**Launch:** `python api.py`
**API:** http://localhost:8000
**Docs:** http://localhost:8000/docs

### Detailed UI Documentation

See [UI_README.md](UI_README.md) for comprehensive documentation including:
- Installation and setup
- Interface features and usage
- API endpoints and examples
- Troubleshooting guide
- Deployment instructions

## Usage Examples

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

- [x] Implement data collection pipelines
- [x] Build feature extraction modules
- [x] Train baseline models
- [x] Create web UI (Streamlit)
- [x] Create demo interface (Gradio)
- [x] Create web API (FastAPI)
- [ ] Implement deep learning models
- [ ] Add citation network features
- [ ] Deploy as web service
- [ ] Add user authentication
- [ ] Add citation trend analysis
