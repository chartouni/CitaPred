# CitaPred Scripts

This directory contains scripts for data collection and model training.

## Overview

The citation predictor pipeline has two main steps:
1. **Data Collection** - Gather research papers from Semantic Scholar
2. **Model Training** - Train and evaluate citation prediction models

## Scripts

### 1. collect_large_dataset.py

Collects ~2000 research papers from Semantic Scholar across multiple queries.

**Usage:**
```powershell
python scripts\collect_large_dataset.py
```

**What it does:**
- Searches for papers in machine learning, deep learning, NLP, and computer vision
- Collects 500 papers per query (2000 total)
- Filters for papers from 2015-2020 (ensuring citation history)
- Removes duplicates
- Saves to `data/raw/complete_dataset.json`

**Configuration:**
Edit the script to customize:
- `QUERIES` - Search terms (currently: ML, DL, NLP, CV)
- `PAPERS_PER_QUERY` - Papers per query (currently: 500)
- `MIN_YEAR` / `MAX_YEAR` - Year range (currently: 2015-2020)

**Expected runtime:** 20-40 minutes (includes rate limiting)

### 2. train_improved_model.py

Trains citation prediction models with improved features and evaluation.

**Usage:**
```powershell
python scripts\train_improved_model.py
```

**Prerequisites:**
- Must run `collect_large_dataset.py` first
- Requires `data/raw/complete_dataset.json` to exist

**What it does:**
- Loads collected dataset
- Performs temporal validation (train on old papers, test on new)
- Applies log transformation to citation counts (reduces skewness)
- Extracts enhanced features (including author metrics)
- Trains Linear and Random Forest models
- Compares model performance
- Shows feature importance
- Displays stratified metrics by citation count

**Key Improvements over basic_usage.py:**
1. **Log transformation** - Handles skewed citation distribution
2. **Temporal validation** - More realistic evaluation
3. **Enhanced features** - Author h-index, citation counts
4. **Stratified metrics** - Shows performance by citation range
5. **Model comparison** - Tests multiple algorithms

## Complete Workflow

### Step 1: Collect Data (20-40 minutes)
```powershell
cd "C:\Users\ici beyrouth\Desktop\CitaPred-claude-citation-predictor-01NMNQaSeWLKfzs7acJgjhmw\CitaPred-claude-citation-predictor-01NMNQaSeWLKfzs7acJgjhmw"

python scripts\collect_large_dataset.py
```

Wait for it to complete. You'll see:
- Progress updates as papers are collected
- Final dataset statistics
- Data saved to `data/raw/complete_dataset.json`

### Step 2: Train Model (2-5 minutes)
```powershell
python scripts\train_improved_model.py
```

You'll see:
- Dataset statistics
- Feature extraction info
- Training progress for each model
- Evaluation metrics (MAE, RMSE, R²)
- Stratified performance by citation count
- Feature importance rankings
- Model comparison summary

### Step 3: Analyze Results

Look for these key metrics:
- **R²** - Should be positive (0.3-0.6 is reasonable for citation prediction)
- **MAE** - Lower is better (50-200 citations is typical)
- **Pearson r** - Correlation (0.5-0.8 is good)

## Expected Performance

With ~2000 papers and improved features:
- **R²**: 0.3 - 0.5 (vs. -20 in basic example)
- **MAE**: 50 - 150 citations (vs. 2755 in basic example)
- **Pearson r**: 0.6 - 0.8 (vs. 0.57 in basic example)

The Random Forest model typically outperforms Linear Regression.

## Troubleshooting

### "Dataset not found" error
**Solution:** Run `collect_large_dataset.py` first

### "No papers collected" error
**Cause:** Internet connection issue or API rate limit
**Solution:** Wait 5 minutes and try again

### "Not enough papers for training"
**Cause:** Collected dataset has < 100 valid papers
**Solution:** Adjust year range or queries in collection script

### Memory errors
**Cause:** TF-IDF text features enabled (commented out by default)
**Solution:** Keep text features disabled or reduce `max_features`

## Customization

### Collect papers from specific venue
Edit `collect_large_dataset.py`:
```python
# Replace QUERIES with venue-specific searches
QUERIES = [
    "venue:NeurIPS",
    "venue:ICML",
    "venue:CVPR",
]
```

### Collect more/fewer papers
Edit `collect_large_dataset.py`:
```python
PAPERS_PER_QUERY = 1000  # Increase for more data
```

### Change temporal split
Edit `train_improved_model.py`:
```python
TRAIN_END_YEAR = 2017  # Train on papers up to 2017
VAL_END_YEAR = 2018    # Validate on 2018 papers
# Test on papers after 2018
```

### Try different models
The improved training script currently tests Linear and Random Forest.

To add XGBoost (requires `pip install xgboost`):
```python
# In train_improved_model.py
from xgboost import XGBRegressor

# Add to train_model function:
elif model_type == 'xgboost':
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
```

## Next Steps

After running these scripts successfully:

1. **Experiment with features** - Add venue impact scores, early citations
2. **Try deep learning** - Use BERT embeddings for title/abstract
3. **Hyperparameter tuning** - Optimize model parameters
4. **Collect more data** - More papers = better predictions
5. **Create visualizations** - Plot predicted vs. actual citations

See the main README for more advanced development ideas.
