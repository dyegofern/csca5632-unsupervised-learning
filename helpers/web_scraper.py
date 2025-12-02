"""
Web Scraper Module
Scrapes brand-specific attributes to differentiate brands from parent companies
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from typing import Optional, Dict, List
from urllib.parse import quote_plus
import warnings
warnings.filterwarnings('ignore')

from config import SCRAPING_CONFIG, SUSTAINABILITY_KEYWORDS, VERBOSE


class BrandScraper:
    """
    Scrapes brand-specific information to create differentiating features
    """

    def __init__(self, config: dict = SCRAPING_CONFIG,
                 keywords: list = SUSTAINABILITY_KEYWORDS,
                 verbose: bool = VERBOSE):
        """
        Initialize brand scraper

        Args:
            config: Scraping configuration (delays, retries, etc.)
            keywords: List of sustainability-related keywords to search for
            verbose: Print progress messages
        """
        self.config = config
        self.keywords = keywords
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config['user_agent']})

    def scrape_brand_data(self, brand_name: str) -> Dict[str, any]:
        """
        Scrape comprehensive brand information

        Args:
            brand_name: Name of the brand to scrape

        Returns:
            Dictionary with scraped brand attributes
        """
        brand_data = {
            'brand_name': brand_name,
            'sector': None,
            'category': None,
            'price_tier': None,
            'sustainability_keyword_count': 0,
            'mission_statement': None,
            'description': None,
            'wikipedia_url': None,
            'scrape_success': False
        }

        try:
            # Try Wikipedia first (most reliable source)
            wiki_data = self._scrape_wikipedia(brand_name)
            if wiki_data['success']:
                brand_data.update(wiki_data['data'])
                brand_data['scrape_success'] = True

            # Analyze text for sustainability keywords
            text_to_analyze = ' '.join([
                str(brand_data.get('description', '')),
                str(brand_data.get('mission_statement', ''))
            ]).lower()

            brand_data['sustainability_keyword_count'] = self._count_keywords(text_to_analyze)

            time.sleep(self.config['delay_between_requests'])

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Error scraping {brand_name}: {str(e)}")

        return brand_data

    def _scrape_wikipedia(self, brand_name: str) -> Dict[str, any]:
        """
        Scrape brand information from Wikipedia

        Args:
            brand_name: Brand name to search

        Returns:
            Dictionary with scraped data and success status
        """
        result = {'success': False, 'data': {}}

        try:
            # Search Wikipedia
            search_url = f"https://en.wikipedia.org/wiki/{quote_plus(brand_name)}"
            response = self._make_request(search_url)

            if response and response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract infobox data (contains structured information)
                infobox = soup.find('table', {'class': 'infobox'})
                if infobox:
                    result['data']['sector'] = self._extract_infobox_field(infobox, ['industry', 'sector'])
                    result['data']['category'] = self._extract_infobox_field(infobox, ['product type', 'products'])

                # Extract first paragraph as description
                first_para = soup.find('p', class_=lambda x: x != 'mw-empty-elt')
                if first_para:
                    result['data']['description'] = first_para.get_text().strip()

                # Extract all text from content
                content = soup.find('div', {'id': 'mw-content-text'})
                if content:
                    result['data']['mission_statement'] = content.get_text()[:1000]  # First 1000 chars

                result['data']['wikipedia_url'] = search_url
                result['success'] = True

                if self.verbose:
                    print(f"  ✓ Wikipedia data found for {brand_name}")

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Wikipedia scraping failed for {brand_name}: {str(e)}")

        return result

    def _extract_infobox_field(self, infobox, field_names: List[str]) -> Optional[str]:
        """
        Extract field from Wikipedia infobox

        Args:
            infobox: BeautifulSoup infobox element
            field_names: List of possible field names to search for

        Returns:
            Field value or None
        """
        for field_name in field_names:
            rows = infobox.find_all('tr')
            for row in rows:
                header = row.find('th')
                if header and field_name.lower() in header.get_text().lower():
                    value_cell = row.find('td')
                    if value_cell:
                        return value_cell.get_text().strip()
        return None

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic

        Args:
            url: URL to request

        Returns:
            Response object or None
        """
        for attempt in range(self.config['max_retries']):
            try:
                response = self.session.get(url, timeout=self.config['timeout'])
                return response
            except requests.RequestException as e:
                if attempt == self.config['max_retries'] - 1:
                    if self.verbose:
                        print(f"  ✗ Request failed after {self.config['max_retries']} attempts: {url}")
                    return None
                time.sleep(self.config['delay_between_requests'])
        return None

    def _count_keywords(self, text: str) -> int:
        """
        Count sustainability keywords in text

        Args:
            text: Text to analyze

        Returns:
            Count of sustainability keywords
        """
        if not text:
            return 0

        count = 0
        for keyword in self.keywords:
            count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text))
        return count

    def scrape_multiple_brands(self, brand_names: List[str]) -> pd.DataFrame:
        """
        Scrape data for multiple brands

        Args:
            brand_names: List of brand names

        Returns:
            DataFrame with scraped brand data
        """
        if self.verbose:
            print(f"Scraping data for {len(brand_names)} brands...")

        scraped_data = []
        for i, brand_name in enumerate(brand_names, 1):
            if self.verbose:
                print(f"[{i}/{len(brand_names)}] Scraping: {brand_name}")

            brand_data = self.scrape_brand_data(brand_name)
            scraped_data.append(brand_data)

        df = pd.DataFrame(scraped_data)

        if self.verbose:
            print(f"\n✓ Scraping complete")
            print(f"  Successfully scraped: {df['scrape_success'].sum()}/{len(df)}")
            print(f"  Average keyword count: {df['sustainability_keyword_count'].mean():.2f}")

        return df

    def enrich_brand_dataframe(self, df: pd.DataFrame,
                               brand_column: str = 'brand_name') -> pd.DataFrame:
        """
        Enrich existing DataFrame with scraped brand data

        Args:
            df: DataFrame with brand names
            brand_column: Column name containing brand names

        Returns:
            DataFrame with scraped features added
        """
        if brand_column not in df.columns:
            raise ValueError(f"Column '{brand_column}' not found in DataFrame")

        # Get unique brand names
        unique_brands = df[brand_column].dropna().unique().tolist()

        # Scrape data
        scraped_df = self.scrape_multiple_brands(unique_brands)

        # Merge with original DataFrame
        enriched_df = df.merge(
            scraped_df,
            left_on=brand_column,
            right_on='brand_name',
            how='left'
        )

        return enriched_df


class ManualBrandEnrichment:
    """
    Helper class for manual brand data enrichment
    Use when web scraping is not feasible or for validation
    """

    @staticmethod
    def create_template(brands: List[str], output_path: str) -> pd.DataFrame:
        """
        Create a CSV template for manual data entry

        Args:
            brands: List of brand names
            output_path: Where to save the template

        Returns:
            Template DataFrame
        """
        template = pd.DataFrame({
            'brand_name': brands,
            'sector': '',
            'category': '',
            'price_tier': '',  # 'budget', 'mid-range', 'premium', 'luxury'
            'sustainability_keyword_count': 0,
            'mission_statement': '',
            'description': '',
            'notes': ''
        })

        template.to_csv(output_path, index=False)
        print(f"Template created at: {output_path}")
        print("Please fill in the brand information manually")

        return template

    @staticmethod
    def load_manual_data(file_path: str) -> pd.DataFrame:
        """
        Load manually entered brand data

        Args:
            file_path: Path to manually filled CSV

        Returns:
            DataFrame with brand data
        """
        return pd.read_csv(file_path)


if __name__ == "__main__":
    # Example usage
    print("Brand Web Scraper - Example Usage\n")

    # Test with a few sample brands
    sample_brands = ['Dove', 'Patagonia', 'Tesla']

    scraper = BrandScraper()
    results = scraper.scrape_multiple_brands(sample_brands)

    print("\nSample Results:")
    print(results[['brand_name', 'sector', 'sustainability_keyword_count', 'scrape_success']])

    print("\n" + "="*60)
    print("Note: For production use, consider:")
    print("1. Rate limiting and respecting robots.txt")
    print("2. Caching results to avoid re-scraping")
    print("3. Manual validation of scraped data")
    print("4. Alternative: Manual data entry using ManualBrandEnrichment")
    print("="*60)
