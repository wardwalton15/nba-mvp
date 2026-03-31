"""
Basketball Reference scraper for NBA MVP prediction data.

This module handles all web scraping from basketball-reference.com
including MVP awards, player stats, advanced stats, and team stats.
"""

import pandas as pd
import time
import logging
from bs4 import BeautifulSoup
import requests
from typing import List, Dict, Optional
import os
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BasketballReferenceScraper:
    """Scrapes NBA data from Basketball Reference website."""

    BASE_URL_AWARDS = "https://www.basketball-reference.com/awards/awards_"
    BASE_URL_LEAGUES = "https://www.basketball-reference.com/leagues/NBA_"

    def __init__(self, cache_dir: str = "data/raw", delay_seconds: int = 3):
        """
        Initialize the scraper.

        Args:
            cache_dir: Directory to cache scraped data
            delay_seconds: Seconds to wait between requests (rate limiting)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_seconds = delay_seconds

    def _is_cache_fresh(self, cache_file: Path, max_age_hours: Optional[float]) -> bool:
        """Return True if cache file exists and is within max_age_hours (None means no expiry)."""
        if not cache_file.exists():
            return False
        if max_age_hours is None:
            return True
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return age < timedelta(hours=max_age_hours)

    def _scrape_table(self, url: str, table_id: str = None) -> pd.DataFrame:
        """
        Generic method to scrape a table from a URL.

        Args:
            url: URL to scrape
            table_id: Optional HTML table ID to find specific table

        Returns:
            DataFrame with scraped data
        """
        try:
            logger.info(f"Scraping {url}")
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.content

            soup = BeautifulSoup(html_content, "html.parser")

            # Find table by ID if specified, otherwise find first table
            if table_id:
                table = soup.find('table', id=table_id)
            else:
                table = soup.find('table')

            if not table:
                logger.warning(f"No table found at {url}")
                return pd.DataFrame()

            # Extract table data
            table_data = []
            rows = table.find_all('tr')
            for row in rows:
                row_data = []
                cells = row.find_all(['td', 'th'])
                for cell in cells:
                    cell_text = cell.get_text(strip=True)
                    row_data.append(cell_text)
                table_data.append(row_data)

            # Create DataFrame (skip first row which is usually header)
            df = pd.DataFrame(table_data[1:])

            return df

        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            return pd.DataFrame()

    def scrape_mvp_awards(self, years: List[int], max_age_hours: Optional[float] = None) -> pd.DataFrame:
        """
        Scrape MVP awards data for given years.

        Args:
            years: List of years to scrape
            max_age_hours: If set, re-scrape if cache file is older than this many hours

        Returns:
            DataFrame with MVP voting data
        """
        cache_file = self.cache_dir / f"mvp_awards_{min(years)}_{max(years)}.csv"

        # Check cache
        if self._is_cache_fresh(cache_file, max_age_hours):
            logger.info(f"Loading MVP awards from cache: {cache_file}")
            return pd.read_csv(cache_file)

        all_data = []

        for year in years:
            url = f"{self.BASE_URL_AWARDS}{year}.html"
            df = self._scrape_table(url)

            if not df.empty:
                df['Year'] = year
                all_data.append(df)

            time.sleep(self.delay_seconds)

        if not all_data:
            return pd.DataFrame()

        # Combine all years
        final_df = pd.concat(all_data, ignore_index=True)

        # Save to cache
        final_df.to_csv(cache_file, index=False)
        logger.info(f"Saved MVP awards to cache: {cache_file}")

        return final_df

    def scrape_player_stats(self, years: List[int], max_age_hours: Optional[float] = None) -> pd.DataFrame:
        """
        Scrape per-game player statistics for given years.

        Args:
            years: List of years to scrape
            max_age_hours: If set, re-scrape if cache file is older than this many hours

        Returns:
            DataFrame with per-game player stats
        """
        cache_file = self.cache_dir / f"player_stats_{min(years)}_{max(years)}.csv"

        # Check cache
        if self._is_cache_fresh(cache_file, max_age_hours):
            logger.info(f"Loading player stats from cache: {cache_file}")
            return pd.read_csv(cache_file)

        all_data = []

        for year in years:
            url = f"{self.BASE_URL_LEAGUES}{year}_per_game.html"
            df = self._scrape_table(url)

            if not df.empty:
                df['Year'] = year
                all_data.append(df)

            time.sleep(self.delay_seconds)

        if not all_data:
            return pd.DataFrame()

        # Combine all years
        final_df = pd.concat(all_data, ignore_index=True)

        # Save to cache
        final_df.to_csv(cache_file, index=False)
        logger.info(f"Saved player stats to cache: {cache_file}")

        return final_df

    def scrape_advanced_stats(self, years: List[int], max_age_hours: Optional[float] = None) -> pd.DataFrame:
        """
        Scrape advanced player statistics for given years.

        Args:
            years: List of years to scrape
            max_age_hours: If set, re-scrape if cache file is older than this many hours

        Returns:
            DataFrame with advanced player stats
        """
        cache_file = self.cache_dir / f"advanced_stats_{min(years)}_{max(years)}.csv"

        # Check cache
        if self._is_cache_fresh(cache_file, max_age_hours):
            logger.info(f"Loading advanced stats from cache: {cache_file}")
            return pd.read_csv(cache_file)

        all_data = []

        for year in years:
            url = f"{self.BASE_URL_LEAGUES}{year}_advanced.html"
            df = self._scrape_table(url)

            if not df.empty:
                df['Year'] = year
                all_data.append(df)

            time.sleep(self.delay_seconds)

        if not all_data:
            return pd.DataFrame()

        # Combine all years
        final_df = pd.concat(all_data, ignore_index=True)

        # Save to cache
        final_df.to_csv(cache_file, index=False)
        logger.info(f"Saved advanced stats to cache: {cache_file}")

        return final_df

    def scrape_team_stats(self, years: List[int], max_age_hours: Optional[float] = None) -> pd.DataFrame:
        """
        Scrape team statistics for given years.

        Args:
            years: List of years to scrape
            max_age_hours: If set, re-scrape if cache file is older than this many hours

        Returns:
            DataFrame with team stats
        """
        cache_file = self.cache_dir / f"team_stats_{min(years)}_{max(years)}.csv"

        # Check cache
        if self._is_cache_fresh(cache_file, max_age_hours):
            logger.info(f"Loading team stats from cache: {cache_file}")
            return pd.read_csv(cache_file)

        all_data = []

        for year in years:
            url = f"{self.BASE_URL_LEAGUES}{year}.html"
            df = self._scrape_table(url, table_id='advanced-team')

            if not df.empty:
                df['Year'] = year
                all_data.append(df)

            time.sleep(self.delay_seconds)

        if not all_data:
            return pd.DataFrame()

        # Combine all years
        final_df = pd.concat(all_data, ignore_index=True)

        # Save to cache
        final_df.to_csv(cache_file, index=False)
        logger.info(f"Saved team stats to cache: {cache_file}")

        return final_df

    def scrape_all_data(self, start_year: int, end_year: int,
                        max_age_hours: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Scrape all data types for a given year range.

        Args:
            start_year: First year to scrape (inclusive)
            end_year: Last year to scrape (inclusive)
            max_age_hours: If set, re-scrape if cache files are older than this many hours

        Returns:
            Dictionary containing all scraped data:
            {
                'mvp_awards': DataFrame,
                'player_stats': DataFrame,
                'advanced_stats': DataFrame,
                'team_stats': DataFrame
            }
        """
        years = list(range(start_year, end_year + 1))

        logger.info(f"Scraping data for years {start_year}-{end_year}")

        data = {
            'mvp_awards': self.scrape_mvp_awards(years, max_age_hours=max_age_hours),
            'player_stats': self.scrape_player_stats(years, max_age_hours=max_age_hours),
            'advanced_stats': self.scrape_advanced_stats(years, max_age_hours=max_age_hours),
            'team_stats': self.scrape_team_stats(years, max_age_hours=max_age_hours)
        }

        logger.info("Data scraping complete")

        return data


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    scraper = BasketballReferenceScraper()
    data = scraper.scrape_all_data(1981, 2024)

    print("\nScraping Summary:")
    for key, df in data.items():
        print(f"{key}: {len(df)} rows")
