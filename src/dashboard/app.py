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
    render_model_metrics,
    render_shap_values,
    render_honorable_mention_card
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

# Plain-language descriptions of model features, shown in "About the Model"
FEATURE_DESCRIPTIONS = {
    'PTS': 'Points per game',
    'team_win_pct': 'Team winning percentage',
    'AST': 'Assists per game',
    'TOV': 'Turnovers per game',
    'NRtg': "Team net rating (point differential per 100 possessions)",
    'FTA': 'Free throw attempts per game',
    'WS/48': 'Win shares per 48 minutes',
    'BPM': 'Box plus/minus',
    'OBPM': 'Offensive box plus/minus',
    'DBPM': 'Defensive box plus/minus',
    'USG%': 'Usage rate',
    'pre_mvps': 'Previous MVP awards won',
}


def load_prior_awards(scraper: BasketballReferenceScraper, year: int) -> pd.DataFrame:
    """Load MVP voting for seasons before `year`, used to count each player's previous MVPs."""
    start = config['data']['historical_start_year']
    end = min(config['data']['historical_end_year'], year - 1)
    return scraper.scrape_mvp_awards(list(range(start, end + 1)))


def render_methodology(feature_importance: pd.DataFrame, metadata: dict):
    """Render the "About the Model" text from the loaded model's own metadata."""
    lines = [
        "### Methodology",
        "",
        "This MVP predictor uses a **Gradient Boosting Regressor** trained on historical "
        f"NBA data from {metadata.get('years_range', 'N/A')}. The model predicts each player's "
        "MVP vote share based on their performance and team success.",
        "",
        "**Features (by importance):**",
    ]
    for _, row in feature_importance.iterrows():
        description = FEATURE_DESCRIPTIONS.get(row['Feature'], row['Feature'])
        lines.append(f"- **{row['Feature']}**: {description} ({row['Importance']:.1%})")

    mse = metadata.get('mse')
    min_games_pct = config['data']['min_games_pct_current']
    lines += [
        "",
        "**Data Source:** Basketball Reference (basketball-reference.com)",
        "",
        f"**Model Performance:** test MSE {mse:.4f}" if isinstance(mse, (int, float))
        else "**Model Performance:** N/A",
        "",
        "**Eligibility Criteria for Current Season:**",
        f"- Played in at least {min_games_pct:.1%} of team games "
        f"(~{round(82 * min_games_pct)} over a full season)",
        f"- Averaged at least {config['data']['min_minutes_current']:.0f} minutes per game",
    ]
    st.markdown("\n".join(lines))


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

    # Scrape data — re-fetch from web if CSV cache is older than 24 hours
    scraper = BasketballReferenceScraper()
    raw_data = scraper.scrape_all_data(year, year, max_age_hours=24)

    # Preprocess
    preprocessor = MVPDataPreprocessor()
    processed_data = preprocessor.process_all(
        raw_data,
        min_games_pct=min_games_pct,
        min_minutes=min_minutes,
        prior_awards=load_prior_awards(scraper, year)
    )

    # Filter to current season
    current_season = processed_data[processed_data['Year'] == year]

    return current_season


@st.cache_data(ttl=3600)
def generate_predictions(current_season_data: pd.DataFrame, top_n: int = 5):
    """
    Generate MVP predictions from current season data.

    Args:
        current_season_data: Preprocessed current season DataFrame
        top_n: Number of top candidates

    Returns:
        Tuple of (top_predictions, feature_importance, metadata, shap_values)
    """
    predictor = MVPPredictor()
    return predictor.predict_with_details(current_season_data, top_n=top_n)


@st.cache_data(ttl=3600)
def load_bubble_candidates(year: int, min_games_pct_current: float, min_minutes: float,
                           eligible_players: tuple):
    """
    Load players who missed the games played threshold but are having strong seasons.

    Args:
        year: Season year
        min_games_pct_current: The strict threshold they missed
        min_minutes: Minimum minutes per game
        eligible_players: Tuple of player names who already qualified (for exclusion)

    Returns:
        DataFrame of bubble candidates with predictions, sorted by predicted share
    """
    logger.info("Loading bubble candidates")

    scraper = BasketballReferenceScraper()
    raw_data = scraper.scrape_all_data(year, year, max_age_hours=24)

    preprocessor = MVPDataPreprocessor()
    all_players = preprocessor.process_all(
        raw_data,
        min_games_pct=0.0,
        min_minutes=min_minutes,
        prior_awards=load_prior_awards(scraper, year)
    )

    current = all_players[all_players['Year'] == year]

    # Keep only players who didn't make the eligible set and have played at least 30% of games
    bubble = current[
        ~current['Player'].isin(eligible_players) &
        (current['pct_of_games'] >= 0.30)
    ]

    if len(bubble) == 0:
        return pd.DataFrame()

    # Generate predictions for bubble candidates
    predictor = MVPPredictor()
    predictions = predictor.predict_current_season(bubble)

    return predictions.head(3)


def main():
    """Main dashboard application."""

    # Title and header
    st.title(config['dashboard']['title'])
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
            top_predictions, feature_importance, metadata, shap_values = generate_predictions(
                current_season,
                top_n=top_n
            )

        # Display top predictions
        st.header(f"Top {len(top_predictions)} MVP Candidates")
        st.markdown("---")

        # Display player cards
        for idx, row in top_predictions.iterrows():
            render_player_card(row, idx)

        # "Just Missed the Cut" section
        st.header("👀 Just Missed the Cut")
        st.markdown(
            "*Players having MVP-caliber seasons who haven't met the minimum "
            f"games played threshold ({config['data']['min_games_pct_current']:.0%} of team games)*"
        )

        eligible_players = tuple(current_season['Player'].tolist())
        bubble_candidates = load_bubble_candidates(
            year=config['data']['current_season'],
            min_games_pct_current=config['data']['min_games_pct_current'],
            min_minutes=config['data']['min_minutes_current'],
            eligible_players=eligible_players
        )

        if len(bubble_candidates) > 0:
            for idx, row in bubble_candidates.iterrows():
                render_honorable_mention_card(row, config['data']['min_games_pct_current'])
        else:
            st.info("No notable players currently below the games threshold.")

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

        # SHAP Values Section
        st.markdown("---")
        render_shap_values(shap_values, top_predictions)

        # Model information
        if show_model_info:
            st.markdown("---")
            st.header("ℹ️ Model Information")
            render_model_metrics(metadata)

        # Methodology section
        with st.expander("📖 About the Model"):
            render_methodology(feature_importance, metadata)

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
