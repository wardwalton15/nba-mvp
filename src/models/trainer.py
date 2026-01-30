"""
Model training module for NBA MVP prediction.

This module handles model training, evaluation, and persistence.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple
import yaml

logger = logging.getLogger(__name__)


class MVPModelTrainer:
    """Handles training and evaluation of MVP prediction models."""

    def __init__(self, config_path: str = "config/config.yaml", model_dir: str = "models/trained"):
        """
        Initialize the trainer.

        Args:
            config_path: Path to configuration file
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_params = self.config['model']['params']
        self.features = self.config['model']['features']
        self.target = self.config['model']['target']

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
        """
        Train a Gradient Boosting Regressor model.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained GradientBoostingRegressor model
        """
        logger.info("Training Gradient Boosting Regressor")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Model parameters: {self.model_params}")

        model = GradientBoostingRegressor(
            n_estimators=self.model_params['n_estimators'],
            learning_rate=self.model_params['learning_rate'],
            max_depth=self.model_params['max_depth'],
            random_state=self.model_params['random_state']
        )

        model.fit(X_train, y_train)

        logger.info("Model training complete")

        return model

    def evaluate_model(self, model: GradientBoostingRegressor,
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model")

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Get feature importance
        feature_importance = dict(zip(X_test.columns, model.feature_importances_))
        feature_importance = {k: float(v) for k, v in sorted(feature_importance.items(),
                                                             key=lambda x: x[1], reverse=True)}

        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'n_test_samples': len(X_test),
            'feature_importance': feature_importance
        }

        logger.info(f"Model MSE: {mse:.6f}")
        logger.info(f"Model MAE: {mae:.6f}")
        logger.info(f"Model RMSE: {rmse:.6f}")

        return metrics

    def save_model(self, model: GradientBoostingRegressor, metadata: Dict,
                   filename: str = None) -> str:
        """
        Save trained model and metadata.

        Args:
            model: Trained model to save
            metadata: Dictionary with model metadata
            filename: Optional custom filename (otherwise uses timestamp)

        Returns:
            Path to saved model file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gbr_model_{timestamp}"

        model_path = self.model_dir / f"{filename}.pkl"
        metadata_path = self.model_dir / f"{filename}_metadata.json"

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Add timestamp to metadata
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['model_params'] = self.model_params
        metadata['features'] = self.features

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

        return str(model_path)

    def load_model(self, filepath: str) -> GradientBoostingRegressor:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {filepath}")

        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        return model

    def load_latest_model(self) -> Tuple[GradientBoostingRegressor, Dict]:
        """
        Load the most recently saved model.

        Returns:
            Tuple of (model, metadata)
        """
        # Find all model files
        model_files = list(self.model_dir.glob("gbr_model_*.pkl"))

        if not model_files:
            raise FileNotFoundError("No trained models found in {self.model_dir}")

        # Get the most recent model
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        # Load model
        model = self.load_model(str(latest_model))

        # Load metadata
        metadata_file = latest_model.with_name(latest_model.stem + "_metadata.json")
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        logger.info(f"Loaded latest model: {latest_model.name}")

        return model, metadata

    def train_and_save(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Complete training pipeline: split data, train, evaluate, and save.

        Args:
            data: Preprocessed DataFrame with features and target
            test_size: Proportion of data to use for testing

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting training pipeline")

        # Prepare data
        X = data[self.features].copy()
        y = data[self.target]

        # Convert features to numeric and fill NaN
        for column in X.columns:
            X[column] = pd.to_numeric(X[column], errors='coerce')
        X = X.fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.model_params['random_state']
        )

        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")

        # Train model
        model = self.train_model(X_train, y_train)

        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)

        # Add training metadata
        metrics['n_train_samples'] = len(X_train)
        metrics['train_test_split'] = test_size
        metrics['years_range'] = f"{int(data['Year'].min())}-{int(data['Year'].max())}"

        # Save model
        self.save_model(model, metrics)

        logger.info("Training pipeline complete")

        return metrics


def main():
    """Main training script."""
    import argparse
    from src.data.scraper import BasketballReferenceScraper
    from src.data.preprocessor import MVPDataPreprocessor

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train NBA MVP prediction model')
    parser.add_argument('--start-year', type=int, default=1981,
                       help='Start year for training data')
    parser.add_argument('--end-year', type=int, default=2024,
                       help='End year for training data')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')

    args = parser.parse_args()

    logger.info(f"Training model on data from {args.start_year} to {args.end_year}")

    # Scrape data
    scraper = BasketballReferenceScraper()
    raw_data = scraper.scrape_all_data(args.start_year, args.end_year)

    # Preprocess data
    preprocessor = MVPDataPreprocessor()
    processed_data = preprocessor.process_all(
        raw_data,
        min_games_pct=0.5,
        min_minutes=None
    )

    # Filter to historical data only (exclude current season for training)
    historical_data = processed_data[processed_data['Year'] <= args.end_year]

    logger.info(f"Training on {len(historical_data)} player-seasons")

    # Train model
    trainer = MVPModelTrainer()
    metrics = trainer.train_and_save(historical_data, test_size=args.test_size)

    # Print results
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"\nFeature Importance:")
    for feature, importance in metrics['feature_importance'].items():
        print(f"  {feature:15s}: {importance:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
