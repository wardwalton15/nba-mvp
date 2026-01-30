"""
Data preprocessing and feature engineering for NBA MVP prediction.

This module handles all data cleaning, merging, and feature creation
from raw scraped data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class MVPDataPreprocessor:
    """Handles data preprocessing and feature engineering for MVP prediction."""

    # Team name to abbreviation mapping
    TEAM_MAPPING = {
        'Philadelphia 76ers': 'PHI',
        'Milwaukee Bucks': 'MIL',
        'Boston Celtics': 'BOS',
        'Phoenix Suns': 'PHO',
        'Los Angeles Lakers': 'LAL',
        'San Antonio Spurs': 'SAS',
        'Chicago Bulls': 'CHI',
        'New York Knicks': 'NYK',
        'Indiana Pacers': 'IND',
        'Portland Trail Blazers': 'POR',
        'Houston Rockets': 'HOU',
        'Washington Bullets': 'WSB',
        'Kansas City Kings': 'KCK',
        'Denver Nuggets': 'DEN',
        'Golden State Warriors': 'GSW',
        'San Diego Clippers': 'SDC',
        'Seattle SuperSonics': 'SEA',
        'Atlanta Hawks': 'ATL',
        'Cleveland Cavaliers': 'CLE',
        'New Jersey Nets': 'NJN',
        'Utah Jazz': 'UTA',
        'Detroit Pistons': 'DET',
        'Dallas Mavericks': 'DAL',
        'Los Angeles Clippers': 'LAC',
        'Sacramento Kings': 'SAC',
        'Charlotte Hornets': 'CHA',
        'Miami Heat': 'MIA',
        'Minnesota Timberwolves': 'MIN',
        'Orlando Magic': 'ORL',
        'Toronto Raptors': 'TOR',
        'Vancouver Grizzlies': 'VAN',
        'Washington Wizards': 'WAS',
        'Memphis Grizzlies': 'MEM',
        'New Orleans Hornets': 'NOH',
        'Charlotte Bobcats': 'CHA',
        'New Orleans/Oklahoma City Hornets': 'NOK',
        'Oklahoma City Thunder': 'OKC',
        'Brooklyn Nets': 'BRK',
        'New Orleans Pelicans': 'NOP'
    }

    def __init__(self):
        """Initialize the preprocessor."""
        pass

    def process_mvp_awards(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw MVP awards data.

        Args:
            raw_df: Raw scraped MVP awards DataFrame

        Returns:
            Cleaned DataFrame with MVP voting data
        """
        logger.info("Processing MVP awards data")

        # Check if DataFrame is empty
        if raw_df.empty or len(raw_df) < 2:
            logger.warning("MVP awards data is empty or has insufficient rows")
            return pd.DataFrame(columns=['Rank', 'Player', 'Age', 'Tm', 'Share', 'Year', 'MVP'])

        # Preserve the Year column before resetting headers
        year_values = raw_df['Year'].copy() if 'Year' in raw_df.columns else None

        # Set proper headers (first row contains column names)
        new_header = raw_df.iloc[0]
        df = raw_df[1:].copy()
        df.columns = new_header
        df.columns = df.columns.astype(str)

        # Restore Year column if it existed
        if year_values is not None and len(year_values) > 1:
            df['Year'] = year_values[1:].values

        # Select relevant columns
        df = df[['Rank', 'Player', 'Age', 'Tm', 'Share', 'Year']]

        # Create MVP binary column (1 if Rank==1, else 0)
        df['MVP'] = (df['Rank'] == '1').astype(int)

        logger.info(f"Processed {len(df)} MVP award records")

        return df

    def process_player_stats(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw player per-game statistics.

        Args:
            raw_df: Raw scraped player stats DataFrame

        Returns:
            Cleaned DataFrame with player stats
        """
        logger.info("Processing player per-game stats")

        # Preserve the Year column before resetting headers
        year_values = raw_df['Year'].copy() if 'Year' in raw_df.columns else None

        # Set headers (excluding Year since we'll add it back)
        headers = ['Rk', 'Player', 'Age', 'Tm', 'Pos', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
                   '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
                   'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Notes']
        df = raw_df.iloc[:, :-1].copy()  # Exclude last column (Year)
        df.columns = headers

        # Restore Year column
        if year_values is not None:
            df['Year'] = year_values.values

        # Convert G (games) to numeric for filtering
        df['G'] = pd.to_numeric(df['G'], errors='coerce')

        # Handle traded players: keep row with max games played
        df = self.handle_traded_players(df)

        # Select relevant columns
        df = df[['Player', 'Year', 'Pos', 'Age', 'Tm', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P',
                 '3PA', '3P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
                 'STL', 'BLK', 'TOV', 'PF', 'PTS']]

        # Remove asterisks from player names
        df['Player'] = df['Player'].str.replace('*', '', regex=False)

        logger.info(f"Processed {len(df)} player stat records")

        return df

    def process_advanced_stats(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw advanced player statistics.

        Args:
            raw_df: Raw scraped advanced stats DataFrame

        Returns:
            Cleaned DataFrame with advanced stats
        """
        logger.info("Processing advanced player stats")

        # Preserve the Year column before resetting headers
        year_values = raw_df['Year'].copy() if 'Year' in raw_df.columns else None

        # Set headers (excluding Year since we'll add it back)
        headers = ['Rk', 'Player', 'Age', 'Tm', 'Pos', 'G', 'MP', 'PER', 'TS%', '3PAr', 'FTr',
                   'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'null',
                   'OWS', 'DWS', 'WS', 'WS/48', 'null2', 'OBPM', 'DBPM', 'BPM', 'VORP']
        df = raw_df.iloc[:, :-1].copy()  # Exclude last column (Year)
        df.columns = headers

        # Restore Year column
        if year_values is not None:
            df['Year'] = year_values.values

        # Convert G to numeric
        df['G'] = pd.to_numeric(df['G'], errors='coerce')

        # Handle traded players
        df = self.handle_traded_players(df)

        # Select relevant columns
        df = df[['Player', 'Age', 'USG%', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'Year']]

        logger.info(f"Processed {len(df)} advanced stat records")

        return df

    def process_team_stats(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw team statistics.

        Args:
            raw_df: Raw scraped team stats DataFrame

        Returns:
            Cleaned DataFrame with team stats and abbreviations
        """
        logger.info("Processing team stats")

        # Preserve the Year column before resetting headers
        year_values = raw_df['Year'].copy() if 'Year' in raw_df.columns else None

        # Set headers (excluding Year since we'll add it back)
        headers = ['Rk', 'Team', 'Age', 'W', 'L', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'ORtg',
                   'DRtg', 'NRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'null', 'eFG%', 'TOV%',
                   'ORB%', 'FT/FGA', 'null3', 'eFG%', 'TOV%', 'DRB%', 'FT/FGA', 'null2',
                   'Arena', 'Attend.', 'Attend./G']
        df = raw_df.iloc[:, :-1].copy()  # Exclude last column (Year)
        df.columns = headers

        # Restore Year column
        if year_values is not None:
            df['Year'] = year_values.values

        # Filter out header rows and league average rows
        df = df[(df['Team'] != 'Team') & (df['Team'] != 'League Average')]

        # Select relevant columns
        df = df[['Team', 'Year', 'W', 'L', 'NRtg']]

        # Remove asterisks from team names
        df['Team'] = df['Team'].str.replace('*', '', regex=False)

        # Add team abbreviations
        team_mapping_df = pd.DataFrame({
            'Team Name': list(self.TEAM_MAPPING.keys()),
            'Abbreviation': list(self.TEAM_MAPPING.values())
        })

        df = pd.merge(df, team_mapping_df, left_on='Team', right_on='Team Name', how='left')
        df = df[['Team', 'Abbreviation', 'Year', 'W', 'L', 'NRtg']]

        # Convert NRtg to numeric
        df['NRtg'] = pd.to_numeric(df['NRtg'], errors='coerce')

        logger.info(f"Processed {len(df)} team stat records")

        return df

    def handle_traded_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle players who were traded during the season.
        Keeps the row with the maximum games played.

        Args:
            df: DataFrame with player data

        Returns:
            DataFrame with one row per player per season
        """
        # Group by player, year, age and keep row with max games
        idx = df.groupby(['Player', 'Year', 'Age'])['G'].idxmax()
        filtered_df = df.loc[idx]

        return filtered_df.reset_index(drop=True)

    def merge_all_data(self, player_stats: pd.DataFrame, team_stats: pd.DataFrame,
                      advanced_stats: pd.DataFrame, awards: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all data sources into a single DataFrame.

        Args:
            player_stats: Processed player stats
            team_stats: Processed team stats
            advanced_stats: Processed advanced stats
            awards: Processed MVP awards

        Returns:
            Merged DataFrame with all data
        """
        logger.info("Merging all data sources")

        # Ensure Year and Age columns are integers
        for df in [player_stats, team_stats, advanced_stats, awards]:
            if df.empty:
                continue
            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df.dropna(subset=['Year'], inplace=True)
                df['Year'] = df['Year'].astype(int)
            if 'Age' in df.columns:
                df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
                df['Age'] = df['Age'].fillna(0).astype(int)

        # Merge players with teams
        merged = pd.merge(
            player_stats,
            team_stats,
            left_on=['Tm', 'Year'],
            right_on=['Abbreviation', 'Year'],
            how='left'
        )

        # Merge with awards (only if awards data exists)
        if not awards.empty:
            merged = pd.merge(
                merged,
                awards,
                on=['Player', 'Year', 'Age'],
                how='left'
            )
        else:
            # Add missing award columns with default values
            merged['MVP'] = 0
            merged['Share'] = 0.0
            merged['Rank'] = None

        # Merge with advanced stats
        merged = pd.merge(
            merged,
            advanced_stats,
            on=['Player', 'Year', 'Age'],
            how='left'
        )

        # Fill missing MVP, NRtg, and Share values
        merged[['MVP', 'NRtg', 'Share']] = merged[['MVP', 'NRtg', 'Share']].fillna(0)

        # Drop unnecessary columns (keep Tm for team abbreviation display)
        columns_to_drop = ['FG', 'FG%', '3P', 'FT', 'DRB', 'Abbreviation', 'Rank']
        merged = merged.drop(columns=[col for col in columns_to_drop if col in merged.columns])

        logger.info(f"Merged data contains {len(merged)} records")

        return merged

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for MVP prediction.

        Features created:
        - team_win_pct: Team win percentage
        - pct_of_games: Percentage of team games played
        - pre_mvps: Number of previous MVP awards won

        Args:
            df: Merged DataFrame

        Returns:
            DataFrame with new features
        """
        logger.info("Creating engineered features")

        df = df.copy()

        # Convert W and L to numeric
        df['W'] = pd.to_numeric(df['W'], errors='coerce')
        df['L'] = pd.to_numeric(df['L'], errors='coerce')

        # Team win percentage
        df['team_win_pct'] = np.where(
            pd.isna(df['W']) | pd.isna(df['L']),
            0.5,  # Default to 0.5 if missing
            df['W'] / (df['W'] + df['L'])
        )

        # Percentage of games played
        df['pct_of_games'] = df['G'] / (df['W'] + df['L'])
        median_pct = df['pct_of_games'].median()
        df['pct_of_games'] = df['pct_of_games'].fillna(median_pct)

        # Previous MVPs won (cumulative sum shifted by 1)
        df = df.sort_values(['Player', 'Year']).reset_index(drop=True)
        df['mvp_cumsum'] = df.groupby('Player')['MVP'].cumsum()
        df['pre_mvps'] = df.groupby('Player')['mvp_cumsum'].shift(fill_value=0)

        logger.info("Feature engineering complete")

        return df

    def filter_eligible_players(self, df: pd.DataFrame, min_games_pct: float = 0.5,
                                min_minutes: float = None) -> pd.DataFrame:
        """
        Filter players based on eligibility criteria.

        Args:
            df: DataFrame with all features
            min_games_pct: Minimum percentage of games played
            min_minutes: Minimum minutes per game (optional)

        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering players (min_games_pct={min_games_pct}, min_minutes={min_minutes})")

        df = df.copy()

        # Convert MP to numeric
        df['MP'] = pd.to_numeric(df['MP'], errors='coerce')

        # Apply filters
        df = df[df['pct_of_games'] >= min_games_pct]

        if min_minutes is not None:
            df = df[df['MP'] >= min_minutes]

        logger.info(f"Filtered to {len(df)} eligible player-seasons")

        return df.reset_index(drop=True)

    def prepare_training_data(self, df: pd.DataFrame,
                             features: list,
                             target: str = 'Share') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.

        Args:
            df: DataFrame with all features
            features: List of feature column names
            target: Target column name

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        logger.info(f"Preparing training data with {len(features)} features")

        # Select features
        X = df[features].copy()

        # Convert all features to numeric
        for column in X.columns:
            X[column] = pd.to_numeric(X[column], errors='coerce')

        # Fill missing values with 0
        X = X.fillna(0)

        # Get target
        y = df[target]

        logger.info(f"Training data prepared: {len(X)} samples, {len(features)} features")

        return X, y

    def process_all(self, raw_data: Dict[str, pd.DataFrame],
                   min_games_pct: float = 0.5,
                   min_minutes: float = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.

        Args:
            raw_data: Dictionary of raw DataFrames from scraper
            min_games_pct: Minimum percentage of games played
            min_minutes: Minimum minutes per game (optional)

        Returns:
            Fully processed DataFrame ready for modeling
        """
        logger.info("Starting complete preprocessing pipeline")

        # Process each data source
        awards = self.process_mvp_awards(raw_data['mvp_awards'])
        player_stats = self.process_player_stats(raw_data['player_stats'])
        advanced_stats = self.process_advanced_stats(raw_data['advanced_stats'])
        team_stats = self.process_team_stats(raw_data['team_stats'])

        # Merge all data
        merged = self.merge_all_data(player_stats, team_stats, advanced_stats, awards)

        # Create features
        merged = self.create_features(merged)

        # Filter eligible players
        filtered = self.filter_eligible_players(merged, min_games_pct, min_minutes)

        logger.info("Preprocessing pipeline complete")

        return filtered


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    print("MVPDataPreprocessor module loaded successfully")
