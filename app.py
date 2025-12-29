"""
CitaPred: Citation Prediction System - Streamlit Web Application

A web interface for predicting paper citations using machine learning.
Supports both classification (highly cited vs not) and regression (citation count).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citapred.features.extractor import FeatureExtractor
from citapred.utils.logger import setup_logger

logger = setup_logger()

# Page config
st.set_page_config(
    page_title="CitaPred - Citation Prediction System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class ModelManager:
    """Manage trained models and make predictions."""

    def __init__(self):
        self.models = {}
        self.feature_extractor = None
        self.load_models()

    def load_models(self):
        """Load all available trained models."""
        models_dir = Path("models")

        if not models_dir.exists():
            st.warning("⚠️ Models directory not found. Please train models first.")
            return

        # Look for classification and regression models
        model_types = ['classification', 'regression']
        dataset_types = ['general', 'aub']

        for dataset in dataset_types:
            for model_type in model_types:
                model_path = models_dir / f"{dataset}_{model_type}_model.pkl"

                if model_path.exists():
                    try:
                        self.models[f"{dataset}_{model_type}"] = joblib.load(model_path)
                        logger.info(f"Loaded {dataset} {model_type} model")
                    except Exception as e:
                        logger.error(f"Error loading {model_path}: {e}")

        # Load feature extractor
        self.feature_extractor = FeatureExtractor()

        if not self.models:
            st.error("❌ No trained models found. Please train models first using scripts/train_classification.py")

    def get_available_models(self):
        """Return list of available models."""
        return list(self.models.keys())

    def predict(self, paper_data, model_name):
        """Make prediction for a single paper."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        # Extract features
        df = pd.DataFrame([paper_data])
        features = self.feature_extractor.extract_features(df)

        model = self.models[model_name]

        # Get prediction
        prediction = model.predict(features)[0]

        # Get probability for classification models
        probability = None
        if 'classification' in model_name and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probability = proba[1]  # Probability of being highly cited

        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = features.columns
            feature_importance = dict(zip(feature_names, importance))

        return {
            'prediction': prediction,
            'probability': probability,
            'feature_importance': feature_importance,
            'features': features
        }


@st.cache_resource
def load_model_manager():
    """Load and cache the model manager."""
    return ModelManager()


def render_header():
    """Render the application header."""
    st.markdown('<div class="main-header">📚 CitaPred</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Citation Prediction System for Academic Papers</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with information and settings."""
    st.sidebar.title("ℹ️ About")
    st.sidebar.info("""
    **CitaPred** predicts the citation impact of academic papers using machine learning.

    **Features:**
    - 🎯 Classification: Highly cited vs not
    - 📊 Regression: Citation count prediction
    - 📈 Feature importance analysis
    - 📁 Batch predictions from CSV

    **Models:**
    - General ML: Trained on CS conference papers
    - AUB: Trained on AUB institutional papers
    """)

    st.sidebar.title("⚙️ Settings")

    # Model selection
    model_manager = load_model_manager()
    available_models = model_manager.get_available_models()

    if not available_models:
        st.sidebar.error("No models available")
        return None

    # Group models by type
    classification_models = [m for m in available_models if 'classification' in m]
    regression_models = [m for m in available_models if 'regression' in m]

    model_type = st.sidebar.radio(
        "Prediction Type",
        ["Classification", "Regression"],
        help="Classification predicts highly cited vs not, Regression predicts citation count"
    )

    if model_type == "Classification" and classification_models:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            classification_models,
            format_func=lambda x: x.replace('_', ' ').title()
        )
    elif model_type == "Regression" and regression_models:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            regression_models,
            format_func=lambda x: x.replace('_', ' ').title()
        )
    else:
        st.sidebar.warning(f"No {model_type.lower()} models available")
        return None

    return selected_model


def render_single_prediction(model_manager, selected_model):
    """Render single paper prediction interface."""
    st.header("📝 Single Paper Prediction")

    st.markdown("""
    Enter paper metadata below to predict its citation impact. All fields are optional,
    but more information leads to better predictions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input(
            "Paper Title *",
            placeholder="e.g., Deep Learning for Citation Prediction",
            help="The title of the paper"
        )

        abstract = st.text_area(
            "Abstract",
            placeholder="Enter the paper abstract...",
            height=150,
            help="The paper abstract (improves prediction accuracy)"
        )

        year = st.number_input(
            "Publication Year *",
            min_value=1950,
            max_value=datetime.now().year,
            value=datetime.now().year,
            help="Year the paper was published"
        )

        venue = st.text_input(
            "Venue",
            placeholder="e.g., NeurIPS, ICML, Nature",
            help="Conference or journal name"
        )

    with col2:
        # Author information
        st.subheader("Author Information")

        num_authors = st.number_input(
            "Number of Authors",
            min_value=1,
            max_value=50,
            value=3
        )

        max_author_hindex = st.number_input(
            "Max Author H-Index",
            min_value=0,
            max_value=300,
            value=10,
            help="Highest h-index among all authors"
        )

        mean_author_hindex = st.number_input(
            "Mean Author H-Index",
            min_value=0,
            max_value=300,
            value=8,
            help="Average h-index across all authors"
        )

        total_author_citations = st.number_input(
            "Total Author Citations",
            min_value=0,
            value=1000,
            help="Total citations across all authors"
        )

    st.markdown("---")

    if st.button("🔮 Predict Citation Impact", type="primary", use_container_width=True):
        if not title:
            st.error("⚠️ Please enter a paper title")
            return

        # Create paper data
        paper_data = {
            'title': title,
            'abstract': abstract if abstract else "",
            'year': year,
            'venue': venue if venue else "",
            'authors': [{'name': f'Author {i+1}'} for i in range(num_authors)],
            'citationCount': 0,  # Placeholder
            'referenceCount': 0,  # Placeholder
        }

        # Add author metrics
        if max_author_hindex > 0:
            for i in range(min(num_authors, 1)):
                paper_data['authors'][i]['hIndex'] = max_author_hindex

        with st.spinner("🔄 Extracting features and making prediction..."):
            try:
                result = model_manager.predict(paper_data, selected_model)

                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("🎯 Prediction Results")

                if 'classification' in selected_model:
                    prediction = "Highly Cited ⭐" if result['prediction'] == 1 else "Not Highly Cited"
                    confidence = result['probability'] * 100 if result['probability'] else 50

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", prediction)
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")

                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={'text': "Confidence Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                else:  # Regression
                    predicted_citations = max(0, int(result['prediction']))

                    st.metric("Predicted Citation Count", f"{predicted_citations:,}")

                    # Citation range estimate (±20%)
                    lower = int(predicted_citations * 0.8)
                    upper = int(predicted_citations * 1.2)
                    st.info(f"📊 Estimated range: {lower:,} - {upper:,} citations")

                st.markdown('</div>', unsafe_allow_html=True)

                # Feature importance
                if result['feature_importance']:
                    st.subheader("📊 Top 15 Important Features")

                    # Sort and get top 15
                    importance_sorted = sorted(
                        result['feature_importance'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:15]

                    features = [f[0] for f in importance_sorted]
                    importances = [f[1] for f in importance_sorted]

                    fig = px.bar(
                        x=importances,
                        y=features,
                        orientation='h',
                        labels={'x': 'Importance', 'y': 'Feature'},
                        title='Feature Importance'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                logger.error(f"Prediction error: {e}", exc_info=True)


def render_batch_prediction(model_manager, selected_model):
    """Render batch prediction interface."""
    st.header("📁 Batch Prediction")

    st.markdown("""
    Upload a CSV file containing multiple papers to get predictions for all of them.

    **Required columns:**
    - `title`: Paper title
    - `year`: Publication year

    **Optional columns:**
    - `abstract`: Paper abstract
    - `venue`: Conference/journal name
    - `authors`: Author names (semicolon-separated)
    - `max_author_hindex`: Highest h-index among authors
    - `mean_author_hindex`: Average h-index
    """)

    # Download example template
    st.markdown("### 📥 Download Template")

    example_df = pd.DataFrame({
        'title': ['Example Paper 1', 'Example Paper 2'],
        'year': [2023, 2024],
        'abstract': ['This is an example abstract...', 'Another example...'],
        'venue': ['NeurIPS', 'ICML'],
        'authors': ['John Doe; Jane Smith', 'Alice Johnson'],
        'max_author_hindex': [25, 30],
        'mean_author_hindex': [20, 25]
    })

    csv = example_df.to_csv(index=False)
    st.download_button(
        label="📄 Download CSV Template",
        data=csv,
        file_name="citapred_template.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### 📤 Upload Your Data")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with paper metadata"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Loaded {len(df):,} papers from CSV")

            # Show preview
            with st.expander("👀 Preview Data (first 5 rows)"):
                st.dataframe(df.head())

            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner(f"🔄 Processing {len(df):,} papers..."):
                    predictions = []
                    progress_bar = st.progress(0)

                    for idx, row in df.iterrows():
                        try:
                            # Prepare paper data
                            paper_data = {
                                'title': row.get('title', ''),
                                'abstract': row.get('abstract', ''),
                                'year': row.get('year', 2023),
                                'venue': row.get('venue', ''),
                                'authors': [],
                                'citationCount': 0,
                                'referenceCount': 0,
                            }

                            # Parse authors if provided
                            if 'authors' in row and pd.notna(row['authors']):
                                authors = str(row['authors']).split(';')
                                paper_data['authors'] = [{'name': a.strip()} for a in authors]

                            result = model_manager.predict(paper_data, selected_model)

                            if 'classification' in selected_model:
                                pred_label = "Highly Cited" if result['prediction'] == 1 else "Not Highly Cited"
                                confidence = result['probability'] * 100 if result['probability'] else None

                                predictions.append({
                                    'title': row.get('title', ''),
                                    'prediction': pred_label,
                                    'confidence': f"{confidence:.1f}%" if confidence else "N/A"
                                })
                            else:  # Regression
                                pred_citations = max(0, int(result['prediction']))
                                predictions.append({
                                    'title': row.get('title', ''),
                                    'predicted_citations': pred_citations
                                })

                        except Exception as e:
                            logger.error(f"Error processing row {idx}: {e}")
                            predictions.append({
                                'title': row.get('title', ''),
                                'error': str(e)
                            })

                        progress_bar.progress((idx + 1) / len(df))

                    progress_bar.empty()

                    # Display results
                    st.success(f"✅ Processed {len(predictions):,} papers!")

                    results_df = pd.DataFrame(predictions)
                    st.dataframe(results_df, use_container_width=True)

                    # Download results
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv_results,
                        file_name=f"citapred_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                    # Summary statistics
                    if 'classification' in selected_model:
                        st.subheader("📊 Summary Statistics")

                        highly_cited = sum(1 for p in predictions if p.get('prediction') == 'Highly Cited')
                        total = len(predictions)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Papers", f"{total:,}")
                        with col2:
                            st.metric("Highly Cited", f"{highly_cited:,}")
                        with col3:
                            st.metric("Percentage", f"{highly_cited/total*100:.1f}%")

        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")


def render_model_info(model_manager, selected_model):
    """Render model information and statistics."""
    st.header("ℹ️ Model Information")

    if selected_model not in model_manager.models:
        st.warning("No model selected")
        return

    model = model_manager.models[selected_model]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Details")
        st.write(f"**Type:** {type(model).__name__}")
        st.write(f"**Task:** {'Classification' if 'classification' in selected_model else 'Regression'}")
        st.write(f"**Dataset:** {'AUB Papers' if 'aub' in selected_model else 'General ML Papers'}")

    with col2:
        st.subheader("Model File")
        model_path = Path("models") / f"{selected_model}.pkl"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            st.write(f"**Path:** `{model_path}`")
            st.write(f"**Size:** {size_mb:.2f} MB")

    # Feature information
    if hasattr(model, 'n_features_in_'):
        st.metric("Number of Features", f"{model.n_features_in_:,}")


def main():
    """Main application."""
    render_header()

    # Load model manager
    model_manager = load_model_manager()

    # Sidebar
    selected_model = render_sidebar()

    if not selected_model:
        st.error("❌ No models available. Please train models first.")
        st.info("""
        To train models, run:
        ```bash
        python scripts/train_classification.py
        python scripts/train_regression.py
        ```
        """)
        return

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📁 Batch Prediction", "ℹ️ Model Info"])

    with tab1:
        render_single_prediction(model_manager, selected_model)

    with tab2:
        render_batch_prediction(model_manager, selected_model)

    with tab3:
        render_model_info(model_manager, selected_model)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>CitaPred - Citation Prediction System | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
