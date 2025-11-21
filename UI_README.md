# CitaPred UI/UX Documentation

This document provides instructions for running the CitaPred web interfaces and API.

## Available Interfaces

CitaPred provides **three different interfaces** to choose from:

1. **Streamlit Web App** - Full-featured web application with dashboards
2. **Gradio Interface** - Simple, clean interface for quick demos
3. **FastAPI REST API** - Programmatic access via HTTP endpoints

## Prerequisites

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Optional: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 1. Streamlit Web Application

### Features
- 🔮 Single paper prediction with detailed analysis
- 📊 Model performance dashboard
- 🔍 Paper search from Semantic Scholar
- 📈 Interactive model training
- 📄 Batch predictions via CSV upload
- 📉 Interactive visualizations

### Running Streamlit

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Manual URL Access

If it doesn't open automatically:
```
http://localhost:8501
```

### Streamlit Pages

1. **Predict Citations**
   - Enter paper details manually
   - Fetch from Semantic Scholar
   - Upload CSV for batch predictions

2. **Model Dashboard**
   - View model performance metrics
   - Feature importance charts
   - Model information

3. **Search Papers**
   - Search Semantic Scholar database
   - Analyze citation patterns
   - View paper statistics

4. **Train Model**
   - Train custom models
   - Configure training parameters
   - View training progress and metrics

5. **About**
   - Project information
   - Usage guidelines
   - Documentation

## 2. Gradio Interface

### Features
- 🎯 Simple and intuitive UI
- 🔮 Quick predictions
- 🔍 Paper search
- 📈 Model training
- 📱 Mobile-friendly

### Running Gradio

```bash
python app_gradio.py
```

The interface will be available at `http://localhost:7860`

### Gradio Tabs

1. **Predict Citations** - Enter paper details
2. **Search & Predict** - Search and analyze papers
3. **Train Model** - Train the ML model
4. **About** - Documentation and info

### Gradio Sharing (Optional)

To create a public shareable link:

```python
# Edit app_gradio.py
demo.launch(
    share=True  # Set to True for public link
)
```

This creates a temporary public URL (valid for 72 hours).

## 3. FastAPI REST API

### Features
- 🔌 RESTful API endpoints
- 📚 Auto-generated documentation
- 🔐 CORS enabled
- 📊 JSON responses
- 🚀 High performance

### Running the API

```bash
# Development mode with auto-reload
python api.py

# Or using uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Predict Single Paper
```bash
POST /predict
Content-Type: application/json

{
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models...",
  "year": 2017,
  "venue": "NeurIPS",
  "authorCount": 8,
  "referenceCount": 42,
  "avg_author_h_index": 45.0,
  "avg_author_citations": 15000
}
```

#### Batch Predictions
```bash
POST /batch-predict
Content-Type: application/json

{
  "papers": [
    {
      "title": "Paper 1",
      "abstract": "...",
      "year": 2020,
      ...
    },
    {
      "title": "Paper 2",
      "abstract": "...",
      "year": 2021,
      ...
    }
  ]
}
```

#### Search Papers
```bash
POST /search
Content-Type: application/json

{
  "query": "machine learning",
  "limit": 20
}
```

#### Train Model
```bash
POST /train
Content-Type: application/json

{
  "model_type": "random_forest",
  "search_query": "machine learning",
  "num_papers": 200,
  "test_size": 0.2
}
```

#### Get Model Info
```bash
GET /model/info
```

### Example API Usage (Python)

```python
import requests

# Predict citation count
url = "http://localhost:8000/predict"
data = {
    "title": "My Research Paper",
    "abstract": "This paper presents...",
    "year": 2024,
    "venue": "Nature",
    "authorCount": 5,
    "referenceCount": 35
}

response = requests.post(url, json=data)
result = response.json()

print(f"Predicted citations: {result['predicted_citations']}")
print(f"Impact level: {result['impact_level']}")
```

### Example API Usage (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Research Paper",
    "abstract": "This paper presents...",
    "year": 2024,
    "venue": "Nature",
    "authorCount": 5,
    "referenceCount": 35
  }'
```

## Workflow Guide

### First Time Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a model** (choose one method):

   **Option A: Using Streamlit**
   ```bash
   streamlit run app.py
   # Navigate to "Train Model" page
   ```

   **Option B: Using Gradio**
   ```bash
   python app_gradio.py
   # Go to "Train Model" tab
   ```

   **Option C: Using API**
   ```bash
   curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"model_type": "random_forest", "search_query": "machine learning", "num_papers": 200}'
   ```

3. **Make predictions** using any interface

### Making Predictions

#### Manual Input
1. Launch Streamlit or Gradio
2. Go to "Predict Citations" page/tab
3. Fill in paper details
4. Click "Predict"

#### From Semantic Scholar
1. Launch any interface
2. Go to "Search" or "Search & Predict"
3. Enter paper title or keywords
4. View predictions

#### Batch Predictions (CSV)
1. Prepare CSV file with columns:
   - `title` (required)
   - `abstract` (required)
   - `year`
   - `venue`
   - `authorCount`
   - `referenceCount`

2. Upload via Streamlit "Batch Prediction" option
3. Download results

## Configuration

### Port Configuration

**Streamlit** (default: 8501)
```bash
streamlit run app.py --server.port 8080
```

**Gradio** (default: 7860)
Edit `app_gradio.py`:
```python
demo.launch(server_port=8080)
```

**FastAPI** (default: 8000)
```bash
uvicorn api:app --port 8080
```

### Model Configuration

Models are saved in: `models/citation_predictor.pkl`

To change model type, edit in the training configuration or use:
```python
predictor = CitationPredictor(model_type="gradient_boosting")
```

Available model types:
- `random_forest` (default, best balance)
- `gradient_boosting` (highest accuracy, slower)
- `linear_regression` (fastest, baseline)

## Troubleshooting

### Issue: Model not found

**Solution**: Train a model first using any interface

### Issue: "No papers collected"

**Solution**: Check internet connection and Semantic Scholar API availability

### Issue: Import errors

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Port already in use

**Solution**: Change the port number
```bash
streamlit run app.py --server.port 8502
```

### Issue: Slow predictions

**Solution**: Use a smaller, faster model
```python
predictor = CitationPredictor(model_type="linear_regression")
```

## Performance Tips

1. **Use batch predictions** for multiple papers
2. **Train on domain-specific data** for better accuracy
3. **Use gradient_boosting** for best accuracy
4. **Use linear_regression** for fastest predictions
5. **Cache models** in production environments

## Production Deployment

### Streamlit Cloud
```bash
# Push to GitHub
git push

# Deploy on streamlit.io
# Add app.py as main file
```

### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# For Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# For FastAPI
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t citapred .
docker run -p 8501:8501 citapred
```

## Screenshots & Features

### Streamlit Features
- Modern, clean interface
- Multiple pages with navigation
- Interactive charts with Plotly
- Real-time training progress
- Download predictions as CSV
- Mobile responsive

### Gradio Features
- Tabbed interface
- Markdown formatting
- Built-in examples
- Share functionality
- Simple deployment

### API Features
- RESTful architecture
- Auto-generated docs
- CORS support
- Error handling
- Pydantic validation

## Next Steps

1. Train a model with your domain-specific data
2. Integrate with your research workflow
3. Deploy to production
4. Customize the UI for your needs
5. Add more features (e.g., graph neural networks)

## Support & Documentation

- **Main README**: `README.md`
- **API Docs**: http://localhost:8000/docs (when API is running)
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory

## License

MIT License - See LICENSE file for details
