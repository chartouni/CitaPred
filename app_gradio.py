"""
CitaPred Gradio Interface

A simple Gradio-based interface for CitaPred citation prediction.
"""

import sys
from pathlib import Path
import gradio as gr
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citapred.data.collector import DataCollector
from citapred.data.preprocessor import DataPreprocessor
from citapred.models.predictor import CitationPredictor

# Load model
predictor = CitationPredictor(model_type="random_forest")
model_path = Path("models/citation_predictor.pkl")
model_loaded = False

if model_path.exists():
    try:
        predictor.load(str(model_path))
        model_loaded = True
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


def predict_citations(title, abstract, year, venue, num_authors, num_references, avg_h_index, avg_citations):
    """
    Predict citation count for a paper.

    Args:
        title: Paper title
        abstract: Paper abstract
        year: Publication year
        venue: Publication venue
        num_authors: Number of authors
        num_references: Number of references
        avg_h_index: Average author H-index
        avg_citations: Average author citations

    Returns:
        Formatted prediction string
    """
    try:
        if not title or not abstract:
            return "❌ Error: Title and abstract are required"

        # Create paper data
        paper_data = pd.DataFrame([{
            'title': title,
            'abstract': abstract,
            'year': int(year),
            'venue': venue,
            'authorCount': int(num_authors),
            'referenceCount': int(num_references),
            'avg_author_h_index': float(avg_h_index),
            'avg_author_citations': int(avg_citations)
        }])

        # Make prediction
        prediction = predictor.predict(paper_data)
        predicted_count = int(prediction[0])

        # Calculate confidence interval
        lower_bound = max(0, int(predicted_count * 0.7))
        upper_bound = int(predicted_count * 1.3)

        # Determine impact level
        if predicted_count < 10:
            impact = "🔵 Low Impact"
            emoji = "📄"
        elif predicted_count < 50:
            impact = "🟡 Moderate Impact"
            emoji = "📝"
        elif predicted_count < 200:
            impact = "🟢 High Impact"
            emoji = "📚"
        else:
            impact = "⭐ Exceptional Impact"
            emoji = "🏆"

        # Format result
        result = f"""
## {emoji} Prediction Results

### Predicted Citation Count: **{predicted_count:,}**

**Confidence Range:** {lower_bound:,} - {upper_bound:,} citations

**Impact Level:** {impact}

---

### Interpretation:
"""

        if predicted_count < 10:
            result += "This paper is predicted to have limited citation impact. Consider revising or targeting a more visible venue."
        elif predicted_count < 50:
            result += "This paper is predicted to receive moderate attention in its field. It will likely contribute to ongoing discussions."
        elif predicted_count < 200:
            result += "This paper is predicted to be well-cited and influential in its domain. It addresses important research questions."
        else:
            result += "This paper is predicted to be highly influential and widely cited. It may become a seminal work in its field."

        if not model_loaded:
            result += "\n\n⚠️ **Note:** Using untrained model. Train the model for better predictions."

        return result

    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"


def search_and_predict(query):
    """
    Search for a paper and predict its citations.

    Args:
        query: Search query

    Returns:
        Formatted result string
    """
    try:
        if not query:
            return "❌ Error: Please enter a search query"

        # Search for paper
        collector = DataCollector()
        papers = collector.search_papers(query=query, limit=1)

        if not papers:
            return f"❌ No papers found for query: '{query}'"

        paper = papers[0]

        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_paper_data([paper])

        # Predict
        prediction = predictor.predict(df)
        predicted_count = int(prediction[0])

        # Get paper info
        title = paper.get('title', 'N/A')
        year = paper.get('year', 'N/A')
        authors = len(paper.get('authors', []))
        actual_citations = paper.get('citationCount', 'N/A')

        # Format result
        result = f"""
## 📄 Paper Found

**Title:** {title}
**Year:** {year}
**Authors:** {authors}
**Actual Citations:** {actual_citations}

---

## 🔮 Prediction

**Predicted Citations:** {predicted_count:,}
"""

        if actual_citations and actual_citations != 'N/A':
            error = abs(predicted_count - actual_citations)
            error_pct = (error / actual_citations * 100) if actual_citations > 0 else 0
            result += f"\n**Prediction Error:** {error:.0f} ({error_pct:.1f}%)"

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


def train_model(query, num_papers, model_type):
    """
    Train the citation prediction model.

    Args:
        query: Search query for training data
        num_papers: Number of papers to collect
        model_type: Type of model to train

    Returns:
        Training status message
    """
    global predictor, model_loaded

    try:
        if not query:
            return "❌ Error: Please enter a search query"

        result = f"🚀 Starting training...\n\n"

        # Collect data
        result += f"📥 Collecting {num_papers} papers about '{query}'...\n"
        collector = DataCollector()
        papers = collector.search_papers(query=query, limit=int(num_papers))

        if len(papers) < 10:
            return f"❌ Error: Not enough papers collected ({len(papers)}). Try a different query."

        result += f"✅ Collected {len(papers)} papers\n\n"

        # Preprocess
        result += "🔧 Preprocessing data...\n"
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_paper_data(papers)

        # Filter papers with citations
        df = df[df['citationCount'].notna() & (df['citationCount'] > 0)]

        if len(df) < 10:
            return f"❌ Error: Not enough papers with citation data ({len(df)})"

        result += f"✅ Preprocessed {len(df)} papers\n\n"

        # Split data
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        result += f"📊 Train set: {len(train_df)} papers | Test set: {len(test_df)} papers\n\n"

        # Train
        result += f"🎓 Training {model_type} model...\n"
        predictor = CitationPredictor(model_type=model_type)
        predictor.train(train_df)

        # Evaluate
        metrics = predictor.evaluate(test_df)

        result += "✅ Training complete!\n\n"
        result += "## 📊 Model Performance\n\n"
        result += f"- **R² Score:** {metrics.get('r2', 0):.3f}\n"
        result += f"- **MAE:** {metrics.get('mae', 0):.1f}\n"
        result += f"- **RMSE:** {metrics.get('rmse', 0):.1f}\n"
        result += f"- **Correlation:** {metrics.get('correlation', 0):.3f}\n\n"

        # Save model
        model_path = Path("models/citation_predictor.pkl")
        model_path.parent.mkdir(exist_ok=True)
        predictor.save(str(model_path))

        model_loaded = True

        result += "💾 Model saved successfully!"

        return result

    except Exception as e:
        return f"❌ Training error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="CitaPred - Citation Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 📚 CitaPred: Research Paper Citation Predictor

    Predict the citation impact of research papers using machine learning.
    """)

    with gr.Tabs():

        # Tab 1: Manual Prediction
        with gr.Tab("🔮 Predict Citations"):
            gr.Markdown("### Enter paper details to predict citation count")

            with gr.Row():
                with gr.Column():
                    title_input = gr.Textbox(
                        label="Paper Title *",
                        placeholder="e.g., Attention Is All You Need",
                        lines=2
                    )
                    abstract_input = gr.Textbox(
                        label="Abstract *",
                        placeholder="Enter the paper abstract...",
                        lines=5
                    )

                with gr.Column():
                    year_input = gr.Number(
                        label="Publication Year",
                        value=2024,
                        precision=0
                    )
                    venue_input = gr.Textbox(
                        label="Venue/Journal",
                        placeholder="e.g., NeurIPS, Nature"
                    )
                    authors_input = gr.Number(
                        label="Number of Authors",
                        value=3,
                        precision=0
                    )
                    refs_input = gr.Number(
                        label="Number of References",
                        value=30,
                        precision=0
                    )

            with gr.Row():
                h_index_input = gr.Number(
                    label="Average H-Index of Authors",
                    value=10.0
                )
                citations_input = gr.Number(
                    label="Average Previous Citations per Author",
                    value=1000,
                    precision=0
                )

            predict_btn = gr.Button("🎯 Predict Citation Count", variant="primary", size="lg")
            prediction_output = gr.Markdown(label="Prediction Results")

            predict_btn.click(
                fn=predict_citations,
                inputs=[
                    title_input, abstract_input, year_input, venue_input,
                    authors_input, refs_input, h_index_input, citations_input
                ],
                outputs=prediction_output
            )

        # Tab 2: Search & Predict
        with gr.Tab("🔍 Search & Predict"):
            gr.Markdown("### Search for a paper from Semantic Scholar and predict its citations")

            search_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter paper title or keywords",
                lines=2
            )
            search_btn = gr.Button("🔍 Search and Predict", variant="primary", size="lg")
            search_output = gr.Markdown(label="Results")

            search_btn.click(
                fn=search_and_predict,
                inputs=search_input,
                outputs=search_output
            )

        # Tab 3: Train Model
        with gr.Tab("📈 Train Model"):
            gr.Markdown("### Train or retrain the citation prediction model")

            with gr.Row():
                with gr.Column():
                    train_query = gr.Textbox(
                        label="Search Query for Training Data",
                        value="machine learning",
                        placeholder="e.g., machine learning, neural networks"
                    )
                    num_papers_input = gr.Slider(
                        label="Number of Papers",
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=50
                    )

                with gr.Column():
                    model_type_input = gr.Dropdown(
                        label="Model Type",
                        choices=["random_forest", "gradient_boosting", "linear_regression"],
                        value="random_forest"
                    )

            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            train_output = gr.Markdown(label="Training Progress")

            train_btn.click(
                fn=train_model,
                inputs=[train_query, num_papers_input, model_type_input],
                outputs=train_output
            )

        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About CitaPred

            CitaPred is a machine learning system that predicts the citation impact of research papers.

            ### How It Works

            1. **Data Collection**: Gather paper metadata from Semantic Scholar
            2. **Feature Extraction**: Extract relevant features (title, abstract, authors, venue, etc.)
            3. **Model Training**: Train ML models on historical citation data
            4. **Prediction**: Predict future citation counts for new papers

            ### Features

            - 🔮 **Single Paper Prediction**: Enter paper details manually
            - 🔍 **Search & Predict**: Fetch papers from Semantic Scholar
            - 📈 **Model Training**: Train custom models on specific domains
            - 🎯 **High Accuracy**: Uses advanced ML algorithms

            ### Model Types

            - **Random Forest**: Ensemble learning method (default)
            - **Gradient Boosting**: Advanced boosting algorithm
            - **Linear Regression**: Simple baseline model

            ### Citation Impact Levels

            - **Low Impact**: < 10 citations
            - **Moderate Impact**: 10-50 citations
            - **High Impact**: 50-200 citations
            - **Exceptional Impact**: 200+ citations

            ### Usage Tips

            1. Train the model first for better predictions
            2. Provide as much paper information as possible
            3. Use domain-specific training data for specialized fields
            4. Consider the confidence range when interpreting results

            ### Data Source

            Papers are collected from [Semantic Scholar](https://www.semanticscholar.org/),
            a free AI-powered research tool for scientific literature.

            ---

            **Version**: 1.0.0
            **License**: MIT
            """)

    gr.Markdown("""
    ---
    <center>
    Made with ❤️ using Gradio and CitaPred
    </center>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
