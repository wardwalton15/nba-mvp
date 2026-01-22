"""
Streamlit dashboard for NBA MVP predictions.

This is the main application file for the interactive dashboard.
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.scraper import BasketballReferenceScraper
from src.data.preprocessor import MVPDataPreprocessor
from src.models.predictor import MVPPredictor
from src.dashboard.components import (
    render_player_card,
    render_feature_importance_chart,
    render_top_predictions_chart,
    render_model_metrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="NBA MVP Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from YAML file."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


config = load_config()


# Data loading functions with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_current_season_data(year: int, min_games_pct: float, min_minutes: float):
    """
    Load and preprocess current season data.

    Args:
        year: Season year
        min_games_pct: Minimum games percentage
        min_minutes: Minimum minutes per game

    Returns:
        Preprocessed DataFrame for current season
    """
    logger.info(f"Loading data for {year} season")

    # Scrape data
    scraper = BasketballReferenceScraper()
    raw_data = scraper.scrape_all_data(year, year)

    # Preprocess
    preprocessor = MVPDataPreprocessor()
    processed_data = preprocessor.process_all(
        raw_data,
        min_games_pct=min_games_pct,
        min_minutes=min_minutes
    )

    # Filter to current season
    current_season = processed_data[processed_data['Year'] == year]

    return current_season


@st.cache_data
def generate_predictions(current_season_data: pd.DataFrame, top_n: int = 5):
    """
    Generate MVP predictions from current season data.

    Args:
        current_season_data: Preprocessed current season DataFrame
        top_n: Number of top candidates

    Returns:
        Tuple of (top_predictions, feature_importance, metadata)
    """
    predictor = MVPPredictor()
    return predictor.predict_with_details(current_season_data, top_n=top_n)


def main():
    """Main dashboard application."""

    # Title and header
    st.title("🏀 NBA MVP Predictor 2025-26")
    st.markdown("*Predicting NBA Most Valuable Player using Machine Learning*")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        show_all_predictions = st.checkbox("Show All Candidates", value=False)
        show_feature_importance = st.checkbox("Show Feature Importance", value=True)
        show_model_info = st.checkbox("Show Model Information", value=False)

    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()

    with col2:
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.caption(f"Last updated: {last_updated}")

    # Load data
    try:
        with st.spinner("Loading current season data..."):
            current_season = load_current_season_data(
                year=config['data']['current_season'],
                min_games_pct=config['data']['min_games_pct_current'],
                min_minutes=config['data']['min_minutes_current']
            )

        if len(current_season) == 0:
            st.warning(f"No player data available for the {config['data']['current_season']} season yet.")
            st.info("Please check back later in the season when sufficient games have been played.")
            return

        # Generate predictions
        with st.spinner("Generating MVP predictions..."):
            top_n = 10 if show_all_predictions else config['dashboard']['top_n']
            top_predictions, feature_importance, metadata = generate_predictions(
                current_season,
                top_n=top_n
            )

        # Display top predictions
        st.header(f"Top {len(top_predictions)} MVP Candidates")
        st.markdown("---")

        # Display player cards
        for idx, row in top_predictions.iterrows():
            render_player_card(row, idx)

        # Visualizations
        st.markdown("---")
        st.header("📊 Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predicted Vote Share Distribution")
            render_top_predictions_chart(top_predictions)

        with col2:
            if show_feature_importance:
                st.subheader("Feature Importance")
                render_feature_importance_chart(feature_importance)
            else:
                st.info("Enable 'Show Feature Importance' in the sidebar to view feature importance.")

        # Model information
        if show_model_info:
            st.markdown("---")
            st.header("ℹ️ Model Information")
            render_model_metrics(metadata)

        # Methodology section
        with st.expander("📖 About the Model"):
            st.markdown("""
            ### Methodology

            This MVP predictor uses a **Gradient Boosting Regressor** trained on historical NBA data from 1981-2024.
            The model predicts each player's MVP vote share based on their performance and team success.

            **Key Features:**
            - **Points (PTS)**: Points per game - most important feature (41.4%)
            - **Team Win %**: Team winning percentage - second most important (36.1%)
            - **Assists (AST)**: Assists per game
            - **Free Throw Attempts (FTA)**: Indicates how often player gets to the line
            - **Net Rating (NRtg)**: Team's point differential per 100 possessions
            - **Previous MVPs**: Number of previous MVP awards won
            - **Turnovers (TOV)**: Turnovers per game
            - **Advanced Stats**: WS/48, OBPM, DBPM, USG%

            **Data Source:** Basketball Reference (basketball-reference.com)

            **Model Performance:** MSE < 0.003 (very accurate)

            **Eligibility Criteria for Current Season:**
            - Played in at least 79% of team games
            - Averaged at least 20 minutes per game
            """)

        # Footer
        st.markdown("---")
        st.caption("Data source: Basketball-Reference.com | Model: Gradient Boosting Regressor")
        st.caption("This is a predictive model for educational and entertainment purposes.")

    except FileNotFoundError as e:
        st.error("⚠️ Model not found. Please train the model first.")
        st.info("Run: `python -m src.models.trainer` to train the model.")
        logger.error(f"Model file not found: {e}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Dashboard error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
