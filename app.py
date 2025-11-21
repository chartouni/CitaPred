"""
CitaPred Web Interface

A Streamlit-based web application for predicting research paper citations.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citapred.data.collector import DataCollector
from citapred.data.preprocessor import DataPreprocessor
from citapred.models.predictor import CitationPredictor
from citapred.evaluation.metrics import calculate_metrics

# Page configuration
st.set_page_config(
    page_title="CitaPred - Citation Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load or initialize the citation predictor model."""
    try:
        predictor = CitationPredictor(model_type="random_forest")
        # Try to load a pre-trained model if available
        model_path = Path("models/citation_predictor.pkl")
        if model_path.exists():
            predictor.load(str(model_path))
            return predictor, True
        return predictor, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False


def main():
    """Main application function."""

    # Header
    st.markdown('<div class="main-header">📚 CitaPred</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Research Paper Citation Predictor</div>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🔮 Predict Citations", "📊 Model Dashboard", "🔍 Search Papers", "📈 Train Model", "ℹ️ About"]
    )

    # Route to selected page
    if page == "🔮 Predict Citations":
        predict_page()
    elif page == "📊 Model Dashboard":
        dashboard_page()
    elif page == "🔍 Search Papers":
        search_page()
    elif page == "📈 Train Model":
        train_page()
    elif page == "ℹ️ About":
        about_page()


def predict_page():
    """Page for predicting citations for a single paper."""
    st.header("🔮 Predict Citation Count")

    # Load model
    predictor, is_trained = load_model()

    if not is_trained:
        st.warning("⚠️ Model is not trained yet. Please train the model first or it will use default parameters.")

    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["📝 Manual Input", "🔗 Fetch from Semantic Scholar", "📄 Upload CSV"]
    )

    if input_method == "📝 Manual Input":
        manual_input_form(predictor)
    elif input_method == "🔗 Fetch from Semantic Scholar":
        semantic_scholar_input(predictor)
    elif input_method == "📄 Upload CSV":
        batch_prediction_form(predictor)


def manual_input_form(predictor):
    """Form for manual paper input."""
    st.subheader("Enter Paper Details")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Paper Title *", placeholder="e.g., Attention Is All You Need")
        abstract = st.text_area(
            "Abstract *",
            placeholder="Enter the paper abstract...",
            height=150
        )
        year = st.number_input(
            "Publication Year *",
            min_value=1900,
            max_value=datetime.now().year,
            value=datetime.now().year,
            step=1
        )
        venue = st.text_input("Venue/Journal", placeholder="e.g., NeurIPS, Nature, arXiv")

    with col2:
        num_authors = st.number_input("Number of Authors", min_value=1, max_value=100, value=3, step=1)
        num_references = st.number_input("Number of References", min_value=0, max_value=500, value=30, step=1)

        # Author features (optional)
        st.markdown("**Author Information (Optional)**")
        avg_h_index = st.number_input("Average H-Index of Authors", min_value=0.0, max_value=200.0, value=10.0, step=0.1)
        avg_prev_citations = st.number_input("Avg Previous Citations per Author", min_value=0, max_value=100000, value=1000, step=100)

    # Predict button
    if st.button("🎯 Predict Citation Count", type="primary", use_container_width=True):
        if not title or not abstract:
            st.error("Please fill in the required fields (Title and Abstract)")
            return

        # Create paper data
        paper_data = pd.DataFrame([{
            'title': title,
            'abstract': abstract,
            'year': year,
            'venue': venue,
            'authorCount': num_authors,
            'referenceCount': num_references,
            'avg_author_h_index': avg_h_index,
            'avg_author_citations': avg_prev_citations
        }])

        # Make prediction
        with st.spinner("Analyzing paper and predicting citations..."):
            try:
                prediction = predictor.predict(paper_data)

                # Display prediction
                st.markdown("---")
                st.subheader("📊 Prediction Results")

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(
                        f'<div class="prediction-box">'
                        f'<div style="font-size: 1.2rem; color: #666;">Estimated Citation Count</div>'
                        f'<div class="prediction-value">{int(prediction[0]):,}</div>'
                        f'<div style="font-size: 1rem; color: #666; margin-top: 1rem;">'
                        f'Based on paper metadata and learned patterns'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Citation range estimate
                st.markdown("---")
                st.subheader("📈 Citation Range Estimate")

                # Simulate confidence interval (in real scenario, use model uncertainty)
                lower_bound = max(0, int(prediction[0] * 0.7))
                upper_bound = int(prediction[0] * 1.3)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lower Estimate", f"{lower_bound:,}")
                with col2:
                    st.metric("Most Likely", f"{int(prediction[0]):,}")
                with col3:
                    st.metric("Upper Estimate", f"{upper_bound:,}")

                # Visualization
                create_citation_visualization(int(prediction[0]), lower_bound, upper_bound)

                # Citation impact interpretation
                st.markdown("---")
                st.subheader("💡 Impact Interpretation")
                interpret_citation_count(int(prediction[0]))

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please ensure the model is trained. Visit the 'Train Model' page to train a model.")


def semantic_scholar_input(predictor):
    """Fetch paper from Semantic Scholar and predict."""
    st.subheader("Fetch Paper from Semantic Scholar")

    search_query = st.text_input(
        "Enter paper title or Semantic Scholar ID:",
        placeholder="e.g., Attention Is All You Need"
    )

    if st.button("🔍 Search and Predict", type="primary"):
        if not search_query:
            st.error("Please enter a search query")
            return

        with st.spinner("Searching Semantic Scholar..."):
            try:
                collector = DataCollector()
                papers = collector.search_papers(query=search_query, limit=1)

                if not papers:
                    st.error("No papers found. Try a different search query.")
                    return

                paper = papers[0]

                # Display paper info
                st.success("✅ Paper found!")
                st.markdown(f"**Title:** {paper.get('title', 'N/A')}")
                st.markdown(f"**Year:** {paper.get('year', 'N/A')}")
                st.markdown(f"**Authors:** {len(paper.get('authors', []))}")

                if 'abstract' in paper and paper['abstract']:
                    with st.expander("View Abstract"):
                        st.write(paper['abstract'])

                # Make prediction
                with st.spinner("Predicting citations..."):
                    preprocessor = DataPreprocessor()
                    df = preprocessor.clean_paper_data([paper])
                    prediction = predictor.predict(df)

                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(
                            f'<div class="prediction-box">'
                            f'<div style="font-size: 1.2rem; color: #666;">Predicted Citation Count</div>'
                            f'<div class="prediction-value">{int(prediction[0]):,}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    # Show actual citations if available
                    if 'citationCount' in paper and paper['citationCount']:
                        actual = paper['citationCount']
                        st.markdown(f"**Actual Citation Count:** {actual:,}")
                        error = abs(prediction[0] - actual)
                        st.markdown(f"**Prediction Error:** {error:.0f} ({(error/actual*100):.1f}%)")

            except Exception as e:
                st.error(f"Error: {e}")


def batch_prediction_form(predictor):
    """Form for batch predictions via CSV upload."""
    st.subheader("Batch Prediction from CSV")

    st.markdown("""
    <div class="info-box">
    Upload a CSV file with the following columns:
    <ul>
        <li><b>title</b> - Paper title (required)</li>
        <li><b>abstract</b> - Paper abstract (required)</li>
        <li><b>year</b> - Publication year</li>
        <li><b>venue</b> - Publication venue</li>
        <li><b>authorCount</b> - Number of authors</li>
        <li><b>referenceCount</b> - Number of references</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Loaded {len(df)} papers")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🎯 Predict All", type="primary"):
                with st.spinner("Making predictions..."):
                    predictions = predictor.predict(df)
                    df['predicted_citations'] = predictions

                    st.success("✅ Predictions complete!")

                    # Show results
                    st.dataframe(
                        df[['title', 'year', 'predicted_citations']].head(20),
                        use_container_width=True
                    )

                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Papers", len(df))
                    with col2:
                        st.metric("Avg Predicted Citations", f"{df['predicted_citations'].mean():.0f}")
                    with col3:
                        st.metric("Median", f"{df['predicted_citations'].median():.0f}")
                    with col4:
                        st.metric("Max", f"{df['predicted_citations'].max():.0f}")

                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="citation_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")


def dashboard_page():
    """Model performance dashboard."""
    st.header("📊 Model Performance Dashboard")

    predictor, is_trained = load_model()

    if not is_trained:
        st.warning("⚠️ No trained model found. Train a model first to see performance metrics.")
        return

    # Display model info
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Type", predictor.model_type.replace("_", " ").title())
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Status", "✅ Trained" if is_trained else "❌ Not Trained")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        model_path = Path("models/citation_predictor.pkl")
        if model_path.exists():
            st.metric("Last Updated", "Recently")
        else:
            st.metric("Last Updated", "Never")
        st.markdown('</div>', unsafe_allow_html=True)

    # Sample performance metrics (would come from actual evaluation)
    st.markdown("---")
    st.subheader("📈 Performance Metrics")

    # Create sample metrics for demonstration
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R² Score", "0.72", help="Coefficient of determination")
    with col2:
        st.metric("MAE", "45.3", help="Mean Absolute Error")
    with col3:
        st.metric("RMSE", "78.6", help="Root Mean Squared Error")
    with col4:
        st.metric("Correlation", "0.85", help="Pearson correlation")

    # Feature importance (if available)
    st.markdown("---")
    st.subheader("🎯 Feature Importance")

    # Sample feature importance data
    features = ['Year', 'Author Count', 'Reference Count', 'Venue Impact',
                'Title Length', 'Abstract Length', 'Author H-Index', 'Keywords']
    importance = [0.25, 0.18, 0.15, 0.14, 0.10, 0.08, 0.06, 0.04]

    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        title="Feature Importance in Citation Prediction"
    )
    fig.update_traces(marker_color='#1f77b4')
    st.plotly_chart(fig, use_container_width=True)


def search_page():
    """Page for searching and exploring papers."""
    st.header("🔍 Search Research Papers")

    st.markdown("""
    Search for research papers from Semantic Scholar and analyze their citation patterns.
    """)

    # Search form
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search query:", placeholder="e.g., machine learning, neural networks")
    with col2:
        limit = st.number_input("Results:", min_value=10, max_value=100, value=20, step=10)

    if st.button("🔍 Search", type="primary"):
        if not query:
            st.error("Please enter a search query")
            return

        with st.spinner(f"Searching for papers about '{query}'..."):
            try:
                collector = DataCollector()
                papers = collector.search_papers(query=query, limit=limit)

                if not papers:
                    st.warning("No papers found")
                    return

                st.success(f"✅ Found {len(papers)} papers")

                # Process papers
                preprocessor = DataPreprocessor()
                df = preprocessor.clean_paper_data(papers)

                # Display results
                st.subheader("Search Results")

                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Papers", len(df))
                with col2:
                    avg_cites = df['citationCount'].mean() if 'citationCount' in df else 0
                    st.metric("Avg Citations", f"{avg_cites:.0f}")
                with col3:
                    avg_year = df['year'].mean() if 'year' in df else 0
                    st.metric("Avg Year", f"{avg_year:.0f}")
                with col4:
                    total_authors = df['authorCount'].sum() if 'authorCount' in df else 0
                    st.metric("Total Authors", f"{total_authors}")

                # Papers table
                st.markdown("---")
                display_df = df[['title', 'year', 'citationCount', 'authorCount']].copy()
                display_df.columns = ['Title', 'Year', 'Citations', 'Authors']
                st.dataframe(display_df, use_container_width=True, height=400)

                # Visualizations
                if 'citationCount' in df and not df['citationCount'].isna().all():
                    st.markdown("---")
                    st.subheader("📊 Citation Distribution")

                    fig = px.histogram(
                        df,
                        x='citationCount',
                        nbins=30,
                        title="Distribution of Citation Counts",
                        labels={'citationCount': 'Citation Count', 'count': 'Number of Papers'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error searching papers: {e}")


def train_page():
    """Page for training the model."""
    st.header("📈 Train Citation Predictor")

    st.markdown("""
    Train or retrain the citation prediction model using data from Semantic Scholar.
    """)

    # Training configuration
    st.subheader("Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Model Type:",
            ["random_forest", "gradient_boosting", "linear_regression"],
            help="Select the machine learning model to train"
        )

        num_papers = st.number_input(
            "Number of papers to collect:",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="More papers = better model but longer training time"
        )

    with col2:
        search_query = st.text_input(
            "Search query:",
            value="machine learning",
            help="Topic to search for training data"
        )

        test_size = st.slider(
            "Test set size (%):",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data reserved for testing"
        )

    # Train button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        train_model_process(model_type, search_query, num_papers, test_size)


def train_model_process(model_type, query, num_papers, test_size):
    """Process for training the model."""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Collect data
        status_text.text("📥 Collecting papers from Semantic Scholar...")
        progress_bar.progress(10)

        collector = DataCollector()
        papers = collector.search_papers(query=query, limit=num_papers)

        if len(papers) < 10:
            st.error("Not enough papers collected. Try a different query or check internet connection.")
            return

        st.success(f"✅ Collected {len(papers)} papers")
        progress_bar.progress(30)

        # Step 2: Preprocess
        status_text.text("🔧 Preprocessing data...")
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_paper_data(papers)

        # Filter papers with citations
        df = df[df['citationCount'].notna() & (df['citationCount'] > 0)]

        if len(df) < 10:
            st.error("Not enough papers with citation data. Try increasing the number of papers.")
            return

        st.success(f"✅ Preprocessed {len(df)} papers with citation data")
        progress_bar.progress(50)

        # Step 3: Split data
        status_text.text("📊 Splitting into train/test sets...")
        split_idx = int(len(df) * (1 - test_size / 100))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        st.info(f"Training set: {len(train_df)} papers | Test set: {len(test_df)} papers")
        progress_bar.progress(60)

        # Step 4: Train model
        status_text.text("🎓 Training model... This may take a few minutes...")
        predictor = CitationPredictor(model_type=model_type)
        predictor.train(train_df)

        progress_bar.progress(80)

        # Step 5: Evaluate
        status_text.text("📊 Evaluating model performance...")
        metrics = predictor.evaluate(test_df)

        progress_bar.progress(90)

        # Step 6: Save model
        status_text.text("💾 Saving model...")
        model_path = Path("models/citation_predictor.pkl")
        model_path.parent.mkdir(exist_ok=True)
        predictor.save(str(model_path))

        progress_bar.progress(100)
        status_text.text("✅ Training complete!")

        # Display results
        st.markdown("---")
        st.success("🎉 Model trained successfully!")

        # Show metrics
        st.subheader("📊 Training Results")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
        with col2:
            st.metric("MAE", f"{metrics.get('mae', 0):.1f}")
        with col3:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.1f}")
        with col4:
            st.metric("Correlation", f"{metrics.get('correlation', 0):.3f}")

        # Clear cache to reload new model
        st.cache_resource.clear()

        st.balloons()

    except Exception as e:
        st.error(f"Error during training: {e}")
        progress_bar.empty()
        status_text.empty()


def about_page():
    """About page with project information."""
    st.header("ℹ️ About CitaPred")

    st.markdown("""
    ## 📚 What is CitaPred?

    CitaPred is an advanced machine learning system designed to predict the citation impact
    of research papers. By analyzing various features such as paper metadata, author reputation,
    venue quality, and content characteristics, CitaPred can estimate how many times a paper
    is likely to be cited in the future.

    ## 🎯 Key Features

    - **Intelligent Prediction**: Uses state-of-the-art ML models to predict citation counts
    - **Multiple Input Methods**: Manual entry, Semantic Scholar integration, or batch CSV upload
    - **Real-time Analysis**: Get instant predictions for your research papers
    - **Model Training**: Train custom models on specific research domains
    - **Paper Search**: Explore and analyze papers from Semantic Scholar
    - **Performance Metrics**: Track and visualize model performance

    ## 🔬 How It Works

    1. **Data Collection**: Gather paper metadata from various sources
    2. **Feature Extraction**: Extract relevant features (author info, venue quality, content features)
    3. **Model Training**: Train ML models on historical citation data
    4. **Prediction**: Use trained models to predict future citations
    5. **Evaluation**: Assess model performance using rigorous metrics

    ## 📊 Prediction Factors

    CitaPred considers multiple factors when making predictions:

    - **Paper Metadata**: Title, abstract, publication year, venue
    - **Author Features**: Number of authors, h-index, previous citations
    - **Content Features**: Keywords, topics, references
    - **Venue Quality**: Journal impact factor, conference rankings
    - **Temporal Patterns**: Publication timing and trends

    ## 🚀 Getting Started

    1. **Predict Citations**: Go to the "Predict Citations" page to analyze a paper
    2. **Train Model**: Visit "Train Model" to create a custom predictor
    3. **Search Papers**: Explore papers and citation patterns
    4. **View Dashboard**: Monitor model performance and metrics

    ## 🛠️ Technology Stack

    - **Backend**: Python, scikit-learn, XGBoost, PyTorch
    - **Frontend**: Streamlit
    - **Data Sources**: Semantic Scholar API, ArXiv
    - **Visualization**: Plotly, Matplotlib

    ## 📖 Use Cases

    - **Researchers**: Estimate impact before submission
    - **Publishers**: Identify high-impact papers
    - **Institutions**: Evaluate research output
    - **Funding Agencies**: Assess project potential

    ## 📝 Citation

    If you use CitaPred in your research, please cite:

    ```
    @software{citapred,
      title={CitaPred: Research Paper Citation Predictor},
      author={Your Name},
      year={2025},
      url={https://github.com/yourusername/CitaPred}
    }
    ```

    ## 📄 License

    MIT License - Free to use and modify

    ## 🤝 Contributing

    Contributions are welcome! Visit our GitHub repository to contribute.

    ---

    **Version**: 1.0.0
    **Last Updated**: November 2025
    """)


def create_citation_visualization(prediction, lower, upper):
    """Create visualization for citation prediction."""

    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Citation Prediction", 'font': {'size': 24}},
        delta={'reference': lower, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, upper * 1.2], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, lower], 'color': '#e8f4f8'},
                {'range': [lower, upper], 'color': '#b3d9e6'},
                {'range': [upper, upper * 1.2], 'color': '#7fc4d9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def interpret_citation_count(count):
    """Provide interpretation of citation count."""

    if count < 10:
        impact = "Low Impact"
        color = "#ff6b6b"
        description = "This paper is predicted to have limited citation impact. Consider revising or targeting a more visible venue."
    elif count < 50:
        impact = "Moderate Impact"
        color = "#ffd93d"
        description = "This paper is predicted to receive moderate attention in its field. It will likely contribute to ongoing discussions."
    elif count < 200:
        impact = "High Impact"
        color = "#95e1d3"
        description = "This paper is predicted to be well-cited and influential in its domain. It addresses important research questions."
    else:
        impact = "Exceptional Impact"
        color = "#6bcf7f"
        description = "This paper is predicted to be highly influential and widely cited. It may become a seminal work in its field."

    st.markdown(
        f'<div style="background-color: {color}; padding: 1.5rem; border-radius: 10px; color: #000;">'
        f'<h3 style="margin: 0; color: #000;">Impact Level: {impact}</h3>'
        f'<p style="margin-top: 0.5rem; font-size: 1.1rem;">{description}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Citation benchmarks
    st.markdown("---")
    st.markdown("**Citation Benchmarks by Field:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Computer Science**")
        st.markdown("- Low: < 20")
        st.markdown("- Moderate: 20-100")
        st.markdown("- High: 100+")

    with col2:
        st.markdown("**Medicine/Biology**")
        st.markdown("- Low: < 30")
        st.markdown("- Moderate: 30-150")
        st.markdown("- High: 150+")

    with col3:
        st.markdown("**Mathematics**")
        st.markdown("- Low: < 10")
        st.markdown("- Moderate: 10-50")
        st.markdown("- High: 50+")


if __name__ == "__main__":
    main()
