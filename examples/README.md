# CitaPred Examples

This directory contains example scripts demonstrating how to use CitaPred.

## Basic Usage

The `basic_usage.py` script demonstrates the complete pipeline:

```bash
python examples/basic_usage.py
```

This will:
1. Collect sample papers from Semantic Scholar
2. Preprocess the data
3. Train a Random Forest model
4. Evaluate predictions

## Running Examples

Make sure you have installed CitaPred first:

```bash
# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

## Next Steps

After running the basic example, check out the Jupyter notebooks in the `notebooks/` directory for more detailed explorations:

- Data exploration and analysis
- Feature engineering experiments
- Model comparison
- Hyperparameter tuning
