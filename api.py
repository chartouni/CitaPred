"""
CitaPred REST API

FastAPI-based REST API for citation prediction service.
"""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citapred.data.collector import DataCollector
from citapred.data.preprocessor import DataPreprocessor
from citapred.models.predictor import CitationPredictor
from citapred.evaluation.metrics import calculate_metrics

# Initialize FastAPI app
app = FastAPI(
    title="CitaPred API",
    description="REST API for predicting research paper citations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
predictor = None


# Pydantic models for request/response
class PaperInput(BaseModel):
    """Input model for a single paper prediction."""
    title: str = Field(..., description="Paper title", min_length=1)
    abstract: str = Field(..., description="Paper abstract", min_length=10)
    year: int = Field(..., description="Publication year", ge=1900, le=2100)
    venue: Optional[str] = Field(None, description="Publication venue or journal")
    authorCount: Optional[int] = Field(3, description="Number of authors", ge=1)
    referenceCount: Optional[int] = Field(30, description="Number of references", ge=0)
    avg_author_h_index: Optional[float] = Field(10.0, description="Average H-index of authors", ge=0)
    avg_author_citations: Optional[int] = Field(1000, description="Average citations per author", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "year": 2017,
                "venue": "NeurIPS",
                "authorCount": 8,
                "referenceCount": 42,
                "avg_author_h_index": 45.0,
                "avg_author_citations": 15000
            }
        }


class BatchPaperInput(BaseModel):
    """Input model for batch predictions."""
    papers: List[PaperInput] = Field(..., description="List of papers to predict")


class PredictionResponse(BaseModel):
    """Response model for citation prediction."""
    predicted_citations: int = Field(..., description="Predicted citation count")
    confidence_interval: dict = Field(..., description="Lower and upper bounds of prediction")
    impact_level: str = Field(..., description="Impact level category")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_citations": 1250,
                "confidence_interval": {"lower": 875, "upper": 1625},
                "impact_level": "High Impact"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    summary: dict


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    is_trained: bool
    version: str
    last_updated: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(..., description="Search query", min_length=1)
    limit: int = Field(20, description="Maximum number of results", ge=1, le=100)


class TrainingConfig(BaseModel):
    """Training configuration model."""
    model_type: str = Field("random_forest", description="Model type to train")
    search_query: str = Field("machine learning", description="Query for collecting training data")
    num_papers: int = Field(200, description="Number of papers to collect", ge=50, le=1000)
    test_size: float = Field(0.2, description="Test set proportion", ge=0.1, le=0.4)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor
    try:
        predictor = CitationPredictor(model_type="random_forest")
        model_path = Path("models/citation_predictor.pkl")
        if model_path.exists():
            predictor.load(str(model_path))
            print("✅ Model loaded successfully")
        else:
            print("⚠️  No trained model found. Model will use default parameters.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CitaPred API",
        "version": "1.0.0",
        "description": "REST API for predicting research paper citations",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "search": "/search",
            "train": "/train",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "version": "1.0.0"
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_path = Path("models/citation_predictor.pkl")
    last_updated = None
    if model_path.exists():
        timestamp = model_path.stat().st_mtime
        last_updated = datetime.fromtimestamp(timestamp).isoformat()

    return {
        "model_type": predictor.model_type,
        "is_trained": model_path.exists(),
        "version": "1.0.0",
        "last_updated": last_updated
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_citation(paper: PaperInput):
    """
    Predict citation count for a single paper.

    Args:
        paper: Paper metadata including title, abstract, year, etc.

    Returns:
        Prediction with confidence interval and impact level
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        paper_data = pd.DataFrame([paper.dict()])

        # Make prediction
        prediction = predictor.predict(paper_data)
        predicted_count = int(prediction[0])

        # Calculate confidence interval (simplified)
        lower_bound = max(0, int(predicted_count * 0.7))
        upper_bound = int(predicted_count * 1.3)

        # Determine impact level
        if predicted_count < 10:
            impact_level = "Low Impact"
        elif predicted_count < 50:
            impact_level = "Moderate Impact"
        elif predicted_count < 200:
            impact_level = "High Impact"
        else:
            impact_level = "Exceptional Impact"

        return {
            "predicted_citations": predicted_count,
            "confidence_interval": {
                "lower": lower_bound,
                "upper": upper_bound
            },
            "impact_level": impact_level
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(batch: BatchPaperInput):
    """
    Predict citation counts for multiple papers.

    Args:
        batch: List of papers to predict

    Returns:
        List of predictions with summary statistics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        papers_data = pd.DataFrame([paper.dict() for paper in batch.papers])

        # Make predictions
        predictions = predictor.predict(papers_data)

        # Create response for each paper
        responses = []
        for pred in predictions:
            predicted_count = int(pred)
            lower_bound = max(0, int(predicted_count * 0.7))
            upper_bound = int(predicted_count * 1.3)

            if predicted_count < 10:
                impact_level = "Low Impact"
            elif predicted_count < 50:
                impact_level = "Moderate Impact"
            elif predicted_count < 200:
                impact_level = "High Impact"
            else:
                impact_level = "Exceptional Impact"

            responses.append({
                "predicted_citations": predicted_count,
                "confidence_interval": {
                    "lower": lower_bound,
                    "upper": upper_bound
                },
                "impact_level": impact_level
            })

        # Calculate summary statistics
        pred_array = [r["predicted_citations"] for r in responses]
        summary = {
            "total_papers": len(predictions),
            "average_citations": sum(pred_array) / len(pred_array),
            "median_citations": sorted(pred_array)[len(pred_array) // 2],
            "min_citations": min(pred_array),
            "max_citations": max(pred_array)
        }

        return {
            "predictions": responses,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/search")
async def search_papers(query: SearchQuery):
    """
    Search for papers from Semantic Scholar.

    Args:
        query: Search query and limit

    Returns:
        List of papers with metadata
    """
    try:
        collector = DataCollector()
        papers = collector.search_papers(query=query.query, limit=query.limit)

        if not papers:
            return {"papers": [], "count": 0}

        # Clean and structure papers
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_paper_data(papers)

        # Convert to dict
        papers_dict = df.to_dict(orient='records')

        return {
            "papers": papers_dict,
            "count": len(papers_dict),
            "query": query.query
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/train")
async def train_model(config: TrainingConfig):
    """
    Train or retrain the citation prediction model.

    Args:
        config: Training configuration

    Returns:
        Training results and metrics
    """
    global predictor

    try:
        # Collect data
        collector = DataCollector()
        papers = collector.search_papers(query=config.search_query, limit=config.num_papers)

        if len(papers) < 10:
            raise HTTPException(status_code=400, detail="Not enough papers collected for training")

        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_paper_data(papers)

        # Filter papers with citations
        df = df[df['citationCount'].notna() & (df['citationCount'] > 0)]

        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Not enough papers with citation data")

        # Split data
        split_idx = int(len(df) * (1 - config.test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Train model
        predictor = CitationPredictor(model_type=config.model_type)
        predictor.train(train_df)

        # Evaluate
        metrics = predictor.evaluate(test_df)

        # Save model
        model_path = Path("models/citation_predictor.pkl")
        model_path.parent.mkdir(exist_ok=True)
        predictor.save(str(model_path))

        return {
            "status": "success",
            "message": "Model trained successfully",
            "data": {
                "total_papers": len(df),
                "train_size": len(train_df),
                "test_size": len(test_df)
            },
            "metrics": metrics,
            "model_type": config.model_type
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.post("/predict-from-semantic-scholar")
async def predict_from_semantic_scholar(query: str):
    """
    Search for a paper on Semantic Scholar and predict its citations.

    Args:
        query: Paper title or Semantic Scholar ID

    Returns:
        Paper info with citation prediction
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        collector = DataCollector()
        papers = collector.search_papers(query=query, limit=1)

        if not papers:
            raise HTTPException(status_code=404, detail="Paper not found")

        paper = papers[0]

        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_paper_data([paper])

        # Predict
        prediction = predictor.predict(df)
        predicted_count = int(prediction[0])

        # Calculate confidence interval
        lower_bound = max(0, int(predicted_count * 0.7))
        upper_bound = int(predicted_count * 1.3)

        # Determine impact level
        if predicted_count < 10:
            impact_level = "Low Impact"
        elif predicted_count < 50:
            impact_level = "Moderate Impact"
        elif predicted_count < 200:
            impact_level = "High Impact"
        else:
            impact_level = "Exceptional Impact"

        return {
            "paper": {
                "title": paper.get('title'),
                "year": paper.get('year'),
                "abstract": paper.get('abstract', '')[:200] + "..." if paper.get('abstract') else None,
                "actual_citations": paper.get('citationCount')
            },
            "prediction": {
                "predicted_citations": predicted_count,
                "confidence_interval": {
                    "lower": lower_bound,
                    "upper": upper_bound
                },
                "impact_level": impact_level
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
