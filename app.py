"""
CitaPred - Citation Prediction Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from citapred.models.predictor import CitationPredictor
from citapred.evaluation.metrics import calculate_metrics

# Page config
st.set_page_config(
    page_title="CitaPred - Citation Predictor",
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
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Header
st.markdown('<div class="main-header">📚 CitaPred</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Research Paper Citation Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selection
    model_type = st.selectbox(
        "Model Type",
        ["linear", "random_forest", "xgboost", "lightgbm"],
        index=0
    )

    st.markdown("---")
    st.header("📊 About")
    st.info(
        "CitaPred uses machine learning to predict how many citations "
        "a research paper will receive based on its metadata, authors, "
        "and content features."
    )

    st.markdown("---")
    st.markdown("### Features Used:")
    st.markdown("""
    - Paper year
    - Title & abstract length
    - Reference count
    - Author count
    - Author h-index stats
    - And more...
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data", "🎯 Train Model", "🔮 Predict", "📊 Visualizations"])

# Tab 1: Data Loading
with tab1:
    st.header("📂 Load Dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload dataset (JSON or CSV)",
            type=["json", "csv"],
            help="Upload your research paper dataset"
        )

        # Or use existing data
        data_path = Path("data/raw/complete_dataset.json")
        if data_path.exists():
            if st.button("📁 Load Existing Dataset"):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                st.session_state.dataset = pd.DataFrame(data)
                st.success(f"✅ Loaded {len(st.session_state.dataset)} papers from {data_path}")

    # Process uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                st.session_state.dataset = pd.DataFrame(data)
            else:
                st.session_state.dataset = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(st.session_state.dataset)} papers")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    # Display dataset info
    if st.session_state.dataset is not None:
        st.markdown("---")
        st.subheader("📋 Dataset Overview")

        df = st.session_state.dataset

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Papers", len(df))
        with col2:
            if 'citationCount' in df.columns:
                st.metric("Avg Citations", f"{df['citationCount'].mean():.0f}")
        with col3:
            if 'year' in df.columns:
                st.metric("Year Range", f"{df['year'].min()}-{df['year'].max()}")
        with col4:
            if 'abstract' in df.columns:
                pct = (df['abstract'].notna().sum() / len(df) * 100)
                st.metric("Has Abstract", f"{pct:.1f}%")

        # Show data
        st.dataframe(df.head(100), use_container_width=True)

        # Citation statistics
        if 'citationCount' in df.columns:
            st.markdown("---")
            st.subheader("📊 Citation Statistics")

            col1, col2 = st.columns(2)

            with col1:
                stats = df['citationCount'].describe()
                st.dataframe(stats.to_frame('Value'), use_container_width=True)

            with col2:
                fig = px.histogram(
                    df,
                    x='citationCount',
                    nbins=50,
                    title='Citation Distribution',
                    labels={'citationCount': 'Citation Count'}
                )
                fig.update_traces(marker_color='#1f77b4')
                st.plotly_chart(fig, use_container_width=True)

# Tab 2: Train Model
with tab2:
    st.header("🎯 Train Prediction Model")

    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first (see Data tab)")
    else:
        df = st.session_state.dataset

        # Check for required columns
        required_cols = ['citationCount']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            st.info(f"Using model type: **{model_type}**")

            # Split settings
            col1, col2 = st.columns(2)

            with col1:
                split_type = st.radio(
                    "Split Method",
                    ["Random Split", "Temporal Split"],
                    help="Temporal split uses year for splitting"
                )

            with col2:
                if split_type == "Random Split":
                    train_size = st.slider("Training Set Size", 0.5, 0.9, 0.7)
                else:
                    if 'year' in df.columns:
                        years = sorted(df['year'].dropna().unique())
                        split_year = st.select_slider(
                            "Training cutoff year",
                            options=years,
                            value=years[int(len(years) * 0.7)] if len(years) > 0 else years[0]
                        )

            # Train button
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Split data
                        if split_type == "Random Split":
                            train_df = df.sample(frac=train_size, random_state=42)
                            test_df = df.drop(train_df.index)
                        else:
                            train_df = df[df['year'] <= split_year]
                            test_df = df[df['year'] > split_year]

                        st.info(f"📊 Training on {len(train_df)} papers, testing on {len(test_df)} papers")

                        # Initialize and train
                        predictor = CitationPredictor(model_type=model_type)
                        predictor.train(train_df, target_col='citationCount')

                        # Evaluate
                        predictions = predictor.predict(test_df)
                        y_true = test_df['citationCount'].values
                        metrics = calculate_metrics(y_true, predictions.values)

                        # Save to session state
                        st.session_state.predictor = predictor
                        st.session_state.model_trained = True
                        st.session_state.test_predictions = predictions
                        st.session_state.test_true = y_true
                        st.session_state.metrics = metrics

                        st.success("✅ Model trained successfully!")

                        # Display metrics
                        st.markdown("---")
                        st.subheader("📈 Model Performance")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{metrics['r2']:.4f}")
                        with col2:
                            st.metric("MAE", f"{metrics['mae']:.2f}")
                        with col3:
                            st.metric("RMSE", f"{metrics['rmse']:.2f}")
                        with col4:
                            st.metric("Pearson r", f"{metrics['pearson_r']:.4f}")

                        # Prediction vs Actual plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=y_true,
                            y=predictions.values,
                            mode='markers',
                            marker=dict(size=8, opacity=0.6, color='#1f77b4'),
                            name='Predictions'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[y_true.min(), y_true.max()],
                            y=[y_true.min(), y_true.max()],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Perfect Prediction'
                        ))
                        fig.update_layout(
                            title='Predicted vs Actual Citations',
                            xaxis_title='Actual Citations',
                            yaxis_title='Predicted Citations',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Error training model: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # Show existing model status
            if st.session_state.model_trained:
                st.markdown("---")
                st.success(f"✅ Model ready: **{model_type}**")

                if 'metrics' in st.session_state:
                    with st.expander("📊 View Detailed Metrics"):
                        st.json(st.session_state.metrics)

# Tab 3: Make Predictions
with tab3:
    st.header("🔮 Predict Citations for New Papers")

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first (see Train Model tab)")
    else:
        st.success("✅ Model is ready for predictions!")

        prediction_mode = st.radio(
            "Prediction Mode",
            ["Manual Input", "Upload Papers"],
            horizontal=True
        )

        if prediction_mode == "Manual Input":
            st.subheader("📝 Enter Paper Details")

            col1, col2 = st.columns(2)

            with col1:
                title = st.text_input("Paper Title", "Deep Learning for Computer Vision")
                abstract = st.text_area(
                    "Abstract (optional)",
                    "This paper presents a novel deep learning approach...",
                    height=150
                )
                year = st.number_input("Publication Year", 2015, 2025, 2023)

            with col2:
                reference_count = st.number_input("Reference Count", 0, 500, 50)
                author_count = st.number_input("Number of Authors", 1, 50, 3)
                venue = st.text_input("Venue (optional)", "CVPR")

            if st.button("🎯 Predict Citations", type="primary"):
                # Create paper dict
                paper = {
                    'title': title,
                    'abstract': abstract if abstract else None,
                    'year': year,
                    'referenceCount': reference_count,
                    'authors': [{'name': f'Author {i}'} for i in range(author_count)],
                    'venue': venue if venue else None
                }

                # Make prediction
                paper_df = pd.DataFrame([paper])
                prediction = st.session_state.predictor.predict(paper_df)

                # Display result
                st.markdown("---")
                st.success("### 🎉 Prediction Result")

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="background-color: #1f77b4; color: white; padding: 2rem; border-radius: 1rem; text-align: center;">
                        <h2 style="margin: 0; color: white;">Predicted Citations</h2>
                        <h1 style="margin: 1rem 0; font-size: 4rem; color: white;">{int(prediction.values[0])}</h1>
                        <p style="margin: 0; color: white; opacity: 0.9;">citations expected</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show input summary
                st.markdown("---")
                with st.expander("📋 View Input Details"):
                    st.json(paper)

        else:  # Upload Papers
            st.subheader("📤 Upload Papers for Batch Prediction")

            uploaded_predict = st.file_uploader(
                "Upload papers (JSON or CSV)",
                type=["json", "csv"],
                key="predict_upload"
            )

            if uploaded_predict is not None:
                try:
                    if uploaded_predict.name.endswith('.json'):
                        data = json.load(uploaded_predict)
                        papers_df = pd.DataFrame(data)
                    else:
                        papers_df = pd.read_csv(uploaded_predict)

                    st.info(f"📊 Loaded {len(papers_df)} papers for prediction")

                    if st.button("🎯 Predict All", type="primary"):
                        with st.spinner("Making predictions..."):
                            predictions = st.session_state.predictor.predict(papers_df)

                            # Add predictions to dataframe
                            result_df = papers_df.copy()
                            result_df['predicted_citations'] = predictions.values

                            st.success(f"✅ Predicted citations for {len(result_df)} papers!")

                            # Display results
                            st.dataframe(
                                result_df[['title', 'year', 'predicted_citations']].head(50),
                                use_container_width=True
                            )

                            # Download button
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Predictions",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )

                            # Statistics
                            st.markdown("---")
                            st.subheader("📊 Prediction Statistics")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Predicted", f"{predictions.mean():.0f}")
                            with col2:
                                st.metric("Median Predicted", f"{predictions.median():.0f}")
                            with col3:
                                st.metric("Max Predicted", f"{predictions.max():.0f}")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Tab 4: Visualizations
with tab4:
    st.header("📊 Dataset Visualizations")

    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first")
    else:
        df = st.session_state.dataset

        # Citation distribution by year
        if 'year' in df.columns and 'citationCount' in df.columns:
            st.subheader("📅 Citations by Year")

            year_stats = df.groupby('year')['citationCount'].agg(['mean', 'median', 'count']).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=year_stats['year'],
                y=year_stats['mean'],
                name='Mean Citations',
                marker_color='#1f77b4'
            ))
            fig.add_trace(go.Scatter(
                x=year_stats['year'],
                y=year_stats['median'],
                name='Median Citations',
                mode='lines+markers',
                marker_color='#ff7f0e',
                line=dict(width=3)
            ))
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Citation Count',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Author count vs citations
        if 'authors' in df.columns and 'citationCount' in df.columns:
            st.markdown("---")
            st.subheader("👥 Author Count vs Citations")

            df_copy = df.copy()
            df_copy['author_count'] = df_copy['authors'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

            fig = px.scatter(
                df_copy,
                x='author_count',
                y='citationCount',
                title='Author Count vs Citation Count',
                labels={'author_count': 'Number of Authors', 'citationCount': 'Citations'},
                trendline='ols',
                opacity=0.6
            )
            fig.update_traces(marker=dict(size=8, color='#1f77b4'))
            st.plotly_chart(fig, use_container_width=True)

        # Reference count vs citations
        if 'referenceCount' in df.columns and 'citationCount' in df.columns:
            st.markdown("---")
            st.subheader("📚 Reference Count vs Citations")

            fig = px.scatter(
                df,
                x='referenceCount',
                y='citationCount',
                title='Reference Count vs Citation Count',
                labels={'referenceCount': 'Number of References', 'citationCount': 'Citations'},
                trendline='ols',
                opacity=0.6
            )
            fig.update_traces(marker=dict(size=8, color='#2ca02c'))
            st.plotly_chart(fig, use_container_width=True)

        # Top cited papers
        if 'citationCount' in df.columns and 'title' in df.columns:
            st.markdown("---")
            st.subheader("🏆 Top Cited Papers")

            top_papers = df.nlargest(10, 'citationCount')[['title', 'citationCount', 'year']]

            fig = px.bar(
                top_papers,
                y='title',
                x='citationCount',
                orientation='h',
                title='Top 10 Most Cited Papers',
                labels={'citationCount': 'Citations', 'title': ''},
                color='citationCount',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">Built with Streamlit | CitaPred © 2025</p>',
    unsafe_allow_html=True
)
