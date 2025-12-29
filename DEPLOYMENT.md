# CitaPred Deployment Guide

This guide explains how to deploy and run the CitaPred web application.

## 🚀 Quick Start

### 1. Install Dependencies

First, ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

### 2. Train Models (if not already done)

Before running the web app, you need trained models. Train both classification and regression models:

```bash
# Train classification model
python scripts/train_classification.py

# Train regression model (optional, but recommended)
python scripts/train_regression.py
```

This will create model files in the `models/` directory:
- `general_classification_model.pkl` - Classification model trained on general ML papers
- `general_regression_model.pkl` - Regression model trained on general ML papers
- `aub_classification_model.pkl` - Classification model trained on AUB papers (if you have AUB data)
- `aub_regression_model.pkl` - Regression model trained on AUB papers (if you have AUB data)

### 3. Run the Web Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📱 Using the Application

### Single Paper Prediction

1. Select your preferred model from the sidebar (Classification or Regression)
2. Navigate to the "📝 Single Prediction" tab
3. Fill in the paper details:
   - **Title** (required): The paper title
   - **Abstract** (optional): Paper abstract for better accuracy
   - **Year** (required): Publication year
   - **Venue** (optional): Conference or journal name
   - **Author metrics**: Number of authors, h-indices, citation counts
4. Click "🔮 Predict Citation Impact"
5. View results:
   - **Classification**: Shows "Highly Cited" or "Not Highly Cited" with confidence score
   - **Regression**: Shows predicted citation count

### Batch Predictions

1. Navigate to the "📁 Batch Prediction" tab
2. Download the CSV template by clicking "📄 Download CSV Template"
3. Fill in your papers data following the template format
4. Upload the filled CSV file
5. Click "🚀 Run Batch Prediction"
6. Download results as CSV with predictions

### Model Information

- Navigate to the "ℹ️ Model Info" tab to view:
  - Model type and architecture
  - Number of features used
  - Model file size and location

## 🎯 Features

### Classification Mode
- **Prediction**: Binary classification (Highly Cited vs Not Highly Cited)
- **Confidence Score**: Probability that paper will be highly cited
- **Threshold**: Top 25% of papers by citation count
- **Visualization**: Confidence gauge, feature importance charts

### Regression Mode
- **Prediction**: Estimated citation count
- **Range Estimate**: ±20% confidence interval
- **Visualization**: Feature importance analysis

### Common Features
- **Feature Importance**: See which features contributed most to the prediction
- **Batch Processing**: Process hundreds of papers at once
- **CSV Export**: Download predictions for further analysis
- **Interactive Charts**: Powered by Plotly for detailed visualization

## 🔧 Configuration

### Model Selection

The app automatically detects available models in the `models/` directory. You can:

1. **Switch between Classification and Regression** using the sidebar radio buttons
2. **Select dataset** (General ML vs AUB) from the dropdown if multiple models exist
3. **Compare predictions** by running the same paper through different models

### Custom Port

To run the app on a different port:

```bash
streamlit run app.py --server.port 8080
```

### Headless Mode (for servers)

To run without opening a browser:

```bash
streamlit run app.py --server.headless true
```

## 📊 Data Format

### Single Prediction Input

Required fields:
- `title`: Paper title (string)
- `year`: Publication year (integer, 1950-2025)

Optional fields for better accuracy:
- `abstract`: Paper abstract (string)
- `venue`: Publication venue (string)
- `max_author_hindex`: Highest h-index among authors (integer, 0-300)
- `mean_author_hindex`: Average h-index (integer, 0-300)
- `total_author_citations`: Total citations across all authors (integer)

### Batch Prediction CSV Format

Example CSV structure:

```csv
title,year,abstract,venue,authors,max_author_hindex,mean_author_hindex
"Deep Learning for Citations",2023,"This paper...","NeurIPS","John Doe; Jane Smith",25,20
"Machine Learning Survey",2024,"A comprehensive...","ICML","Alice Johnson",30,28
```

**Column descriptions:**
- `title` (required): Paper title
- `year` (required): Publication year
- `abstract` (optional): Paper abstract
- `venue` (optional): Conference/journal name
- `authors` (optional): Semicolon-separated author names
- `max_author_hindex` (optional): Maximum h-index
- `mean_author_hindex` (optional): Mean h-index

## 🐛 Troubleshooting

### "No models available" Error

**Cause**: No trained models found in `models/` directory

**Solution**: Train models first:
```bash
python scripts/train_classification.py
```

### "Error making prediction" Message

**Cause**: Missing or invalid paper data

**Solution**:
- Ensure title is provided
- Check year is valid (1950-2025)
- Verify numeric fields are numbers, not text

### Streamlit Installation Issues

**Problem**: `streamlit: command not found`

**Solution**:
```bash
pip install --upgrade streamlit
```

### Slow Predictions

**Cause**: Large models or many papers in batch

**Solutions**:
- For batch predictions, split CSV into smaller files (< 100 papers)
- Use simpler models (LightGBM is faster than deep learning)
- Close other browser tabs to free memory

## 🌐 Production Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy `app.py`

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

Build and run:

```bash
docker build -t citapred .
docker run -p 8501:8501 citapred
```

### AWS/GCP/Azure

Deploy using their respective container services:
- **AWS**: Elastic Beanstalk or ECS
- **GCP**: Cloud Run or App Engine
- **Azure**: App Service or Container Instances

## 📈 Performance Tips

1. **Model Size**: Use LightGBM or XGBoost for faster predictions (vs Neural Networks)
2. **Caching**: Models are cached using `@st.cache_resource` for faster reload
3. **Batch Size**: Process up to 500 papers at once for optimal performance
4. **Feature Extraction**: TF-IDF is computed on-the-fly, so more text = slower processing

## 🔐 Security Considerations

For production deployment:

1. **Enable HTTPS**: Use SSL certificates
2. **Authentication**: Add user authentication if needed (Streamlit doesn't include this by default)
3. **Rate Limiting**: Prevent abuse with rate limits
4. **Input Validation**: Already implemented in the app
5. **File Size Limits**: Default 200MB, adjust with `--server.maxUploadSize`

## 📝 Maintenance

### Updating Models

To update models with new data:

1. Collect new training data
2. Retrain models: `python scripts/train_classification.py`
3. Restart the Streamlit app (it will auto-reload models)

### Monitoring

Monitor app usage with Streamlit's built-in analytics or integrate with:
- Google Analytics
- Mixpanel
- Custom logging in `app.py`

## 🆘 Support

For issues or questions:

1. Check the main [README.md](README.md)
2. Review the [REPORT.md](REPORT.md) for methodology details
3. Check training scripts in `scripts/` directory

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Charts](https://plotly.com/python/)
- [Model Training Scripts](scripts/)
- [Feature Engineering](src/citapred/features/)

---

**Version**: 1.0.0
**Last Updated**: 2025-12-29
**Author**: CitaPred Team
