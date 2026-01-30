"""
Prediction serving module for NBA MVP predictions.

This module handles loading trained models and generating predictions
for current season data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import yaml
import shap
from src.models.trainer import MVPModelTrainer

logger = logging.getLogger(__name__)


class MVPPredictor:
    """Handles loading models and generating MVP predictions."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the predictor.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.features = self.config['model']['features']
        self.trainer = MVPModelTrainer(config_path)
        self.model = None
        self.metadata = None

    def load_latest_model(self):
        """Load the most recently trained model."""
        logger.info("Loading latest model")
        self.model, self.metadata = self.trainer.load_latest_model()
        logger.info(f"Model loaded with MSE: {self.metadata.get('mse', 'N/A')}")

    def predict_current_season(self, current_season_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for current season players.

        Args:
            current_season_data: DataFrame with current season player data

        Returns:
            DataFrame with predictions, sorted by predicted vote share (descending)
        """
        if self.model is None:
            self.load_latest_model()

        logger.info(f"Generating predictions for {len(current_season_data)} players")

        # Prepare features
        X = current_season_data[self.features].copy()

        # Convert to numeric and fill NaN
        for column in X.columns:
            X[column] = pd.to_numeric(X[column], errors='coerce')
        X = X.fillna(0)

        # Generate predictions
        predictions = self.model.predict(X)

        # Create results DataFrame
        results = current_season_data.copy()
        results['Predicted_Share'] = predictions

        # Sort by predicted share (descending)
        results = results.sort_values('Predicted_Share', ascending=False)

        logger.info(f"Predictions generated. Top candidate: {results.iloc[0]['Player']} "
                   f"with {results.iloc[0]['Predicted_Share']:.3f}")

        return results.reset_index(drop=True)

    def rank_top_n(self, predictions: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Get top N MVP candidates.

        Args:
            predictions: DataFrame with predictions
            n: Number of top candidates to return

        Returns:
            DataFrame with top N candidates
        """
        top_n = predictions.head(n).copy()
        top_n['Rank'] = range(1, len(top_n) + 1)

        return top_n

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the loaded model.

        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            self.load_latest_model()

        if self.metadata and 'feature_importance' in self.metadata:
            importance_dict = self.metadata['feature_importance']
            df = pd.DataFrame({
                'Feature': list(importance_dict.keys()),
                'Importance': list(importance_dict.values())
            })
            return df.sort_values('Importance', ascending=False)

        return pd.DataFrame()

    def compute_shap_values(self, data: pd.DataFrame, player_indices: List[int]) -> Dict[str, pd.DataFrame]:
        """
        Compute SHAP values for specific players.

        Args:
            data: Full DataFrame with all player data
            player_indices: List of indices for players to compute SHAP values

        Returns:
            Dictionary mapping player names to their SHAP value DataFrames
        """
        if self.model is None:
            self.load_latest_model()

        logger.info(f"Computing SHAP values for {len(player_indices)} players")

        # Prepare feature matrix for all data (needed for background)
        X_all = data[self.features].copy()
        for column in X_all.columns:
            X_all[column] = pd.to_numeric(X_all[column], errors='coerce')
        X_all = X_all.fillna(0)

        # Use a sample of the data as background for SHAP
        background_size = min(100, len(X_all))
        background = X_all.sample(n=background_size, random_state=42)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model, background)

        # Compute SHAP values for selected players
        X_selected = X_all.iloc[player_indices]
        shap_values = explainer.shap_values(X_selected)

        # Create result dictionary
        result = {}
        for i, idx in enumerate(player_indices):
            player_name = data.iloc[idx]['Player']
            shap_df = pd.DataFrame({
                'Feature': self.features,
                'SHAP_Value': shap_values[i],
                'Feature_Value': X_selected.iloc[i].values
            })
            # Sort by absolute SHAP value
            shap_df['Abs_SHAP'] = shap_df['SHAP_Value'].abs()
            shap_df = shap_df.sort_values('Abs_SHAP', ascending=False)
            result[player_name] = shap_df

        logger.info("SHAP values computed successfully")
        return result

    def predict_with_details(self, current_season_data: pd.DataFrame,
                            top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Dict[str, pd.DataFrame]]:
        """
        Generate predictions with additional details for dashboard display.

        Args:
            current_season_data: DataFrame with current season data
            top_n: Number of top candidates to return

        Returns:
            Tuple of (top_n_predictions, feature_importance, model_metadata, shap_values)
        """
        # Generate predictions
        all_predictions = self.predict_current_season(current_season_data)

        # Get top N
        top_predictions = self.rank_top_n(all_predictions, n=top_n)

        # Get feature importance
        feature_importance = self.get_feature_importance()

        # Compute SHAP values for top 5 players
        top_5_indices = top_predictions.head(5).index.tolist()
        shap_values = self.compute_shap_values(all_predictions, top_5_indices)

        # Model metadata
        metadata = {
            'mse': self.metadata.get('mse', 'N/A'),
            'mae': self.metadata.get('mae', 'N/A'),
            'saved_at': self.metadata.get('saved_at', 'N/A'),
            'years_range': self.metadata.get('years_range', 'N/A'),
            'n_train_samples': self.metadata.get('n_train_samples', 'N/A')
        }

        return top_predictions, feature_importance, metadata, shap_values


def main():
    """Example usage of the predictor."""
    import argparse
    from src.data.scraper import BasketballReferenceScraper
    from src.data.preprocessor import MVPDataPreprocessor

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate NBA MVP predictions')
    parser.add_argument('--year', type=int, default=2025,
                       help='Season year for predictions (default: 2025)')

    args = parser.parse_args()

    logger.info(f"Generating predictions for {args.year} season")

    # Scrape current season data
    scraper = BasketballReferenceScraper()
    raw_data = scraper.scrape_all_data(args.year, args.year)

    # Preprocess data
    preprocessor = MVPDataPreprocessor()

    # Load config to get current season criteria
    import yaml
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    processed_data = preprocessor.process_all(
        raw_data,
        min_games_pct=config['data']['min_games_pct_current'],
        min_minutes=config['data']['min_minutes_current']
    )

    # Filter to current season
    current_season = processed_data[processed_data['Year'] == args.year]

    if len(current_season) == 0:
        logger.warning(f"No player data found for {args.year} season")
        return

    logger.info(f"Found {len(current_season)} eligible players for {args.year}")

    # Generate predictions
    predictor = MVPPredictor()
    top_predictions, feature_importance, metadata = predictor.predict_with_details(
        current_season,
        top_n=10
    )

    # Display results
    print("\n" + "="*60)
    print(f"NBA MVP PREDICTIONS - {args.year} SEASON")
    print("="*60)
    print(f"\nTop 10 Candidates:")
    print("-"*60)

    for idx, row in top_predictions.iterrows():
        print(f"{row['Rank']:2d}. {row['Player']:25s} ({row['Tm']:3s}) - "
              f"Predicted Share: {row['Predicted_Share']:.3f}")
        print(f"    Stats: {row['PTS']:.1f} PPG, {row['AST']:.1f} APG, "
              f"Team Win%: {row['team_win_pct']:.3f}")
        print()

    print("="*60)
    print(f"\nModel Performance:")
    print(f"  MSE: {metadata['mse']}")
    print(f"  Training Period: {metadata['years_range']}")
    print(f"  Training Samples: {metadata['n_train_samples']}")
    print("="*60)


if __name__ == "__main__":
    main()
