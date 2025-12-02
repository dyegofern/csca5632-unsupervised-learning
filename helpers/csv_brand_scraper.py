"""
CSV Brand Scraper
Takes a CSV file with brand names and scrapes ESG/CO2/Pollution data for each brand
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
import json
from datetime import datetime
import os

# Suppress all warnings
warnings.filterwarnings('ignore')

try:
    from ddgs import DDGS
except ImportError:
    # Fallback to old package name if ddgs not installed
    from duckduckgo_search import DDGS

from config import SCRAPING_CONFIG, SUSTAINABILITY_KEYWORDS, VERBOSE, DATA_DIR


class CSVBrandScraper:
    """
    Scraper that takes a CSV with brand names and collects ESG data

    Expected CSV format:
    - Must have a column with brand names (default: 'brand_name')
    - Optional: 'parent_company' column
    - Any other columns will be preserved in output
    """

    def __init__(self, config: dict = SCRAPING_CONFIG,
                 keywords: list = SUSTAINABILITY_KEYWORDS,
                 verbose: bool = VERBOSE,
                 save_progress: bool = False):
        """
        Initialize CSV brand scraper

        Args:
            config: Scraping configuration (delays, retries, etc.)
            keywords: List of sustainability-related keywords to search for
            verbose: Print progress messages
            save_progress: Save data incrementally after each brand
        """
        self.config = config
        self.keywords = keywords
        self.verbose = verbose
        self.save_progress = save_progress
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        # ESG-specific keywords
        self.esg_keywords = [
            'carbon footprint', 'co2 emissions', 'greenhouse gas',
            'pollution', 'environmental impact', 'sustainability report',
            'esg score', 'carbon neutral', 'net zero', 'renewable energy',
            'water usage', 'waste reduction', 'circular economy',
            'environmental compliance', 'climate change'
        ]

        # For incremental saving
        self.progress_file = None

    def scrape_from_csv(self, csv_path: str,
                       brand_column: str = 'brand_name',
                       company_column: str = None) -> pd.DataFrame:
        """
        Main method: Read CSV and scrape ESG data for all brands

        Args:
            csv_path: Path to CSV file with brand names
            brand_column: Name of column containing brand names
            company_column: Name of column containing parent company (optional)

        Returns:
            DataFrame with original data plus ESG information
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"CSV Brand Scraper")
            print(f"{'='*80}\n")

        # Load CSV
        try:
            df = pd.read_csv(csv_path)
            if self.verbose:
                print(f"✓ Loaded CSV: {csv_path}")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            return pd.DataFrame()

        # Validate brand column exists
        if brand_column not in df.columns:
            print(f"❌ Error: Column '{brand_column}' not found in CSV")
            print(f"   Available columns: {list(df.columns)}")
            return pd.DataFrame()

        # Check for company column
        has_company_column = company_column and company_column in df.columns

        if self.verbose:
            print(f"\n📋 Configuration:")
            print(f"  Brand column: '{brand_column}'")
            if has_company_column:
                print(f"  Company column: '{company_column}'")
            else:
                print(f"  Company column: Not specified")
            if self.save_progress:
                print(f"  💾 Incremental saving: ENABLED")
            print()

        # Initialize progress file if needed
        if self.save_progress:
            self._initialize_progress_file(csv_path)

        # Scrape data for each brand (clean brand names to remove newlines, etc.)
        brands_to_scrape = [self._clean_text_for_csv(str(b)) for b in df[brand_column].dropna().tolist()]
        total_brands = len(brands_to_scrape)

        if self.verbose:
            print(f"🔍 Starting scrape for {total_brands} brands...\n")

        # Collect ESG data
        esg_data_list = []

        for i, brand_name in enumerate(brands_to_scrape, 1):
            if self.verbose:
                print(f"[{i}/{total_brands}] Processing: {brand_name}")
                print("-" * 60)

            # Get parent company if available
            company_name = None
            if has_company_column:
                company_name = df.loc[df[brand_column] == brand_name, company_column].iloc[0]
                if pd.notna(company_name):
                    company_name = str(company_name)

            # Scrape brand info and ESG data
            brand_info = self.scrape_brand_info(brand_name, company_name)
            esg_info = self.scrape_esg_data(brand_name, company_name)

            # Combine data
            combined_data = {**brand_info, **esg_info}
            esg_data_list.append(combined_data)

            # Save progress if enabled
            if self.save_progress:
                self._save_incremental_progress(df, esg_data_list, brand_column)
                if self.verbose:
                    print(f"  💾 Progress saved ({i}/{total_brands} brands)")

            time.sleep(self.config['delay_between_requests'])

        # Create DataFrame with ESG data
        esg_df = pd.DataFrame(esg_data_list)

        # Merge with original DataFrame
        result_df = df.merge(
            esg_df,
            left_on=brand_column,
            right_on='brand_name',
            how='left'
        )

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✓ Scraping complete")
            print(f"  Total brands: {total_brands}")
            print(f"  Brands with data: {esg_df['scrape_success'].sum()}")
            print(f"  ESG data found: {esg_df['esg_data_found'].sum()}")
            print(f"{'='*80}\n")

        return result_df

    def scrape_brand_info(self, brand_name: str, parent_company: Optional[str] = None) -> Dict:
        """
        Scrape general brand information using DuckDuckGo and web scraping

        Args:
            brand_name: Brand name
            parent_company: Parent company name (optional)

        Returns:
            Dictionary with brand information
        """
        brand_data = {
            'brand_name': self._clean_text_for_csv(brand_name) if brand_name else None,
            'parent_company': self._clean_text_for_csv(parent_company) if parent_company else None,
            'sector': None,
            'category': None,
            'description': None,
            'website': None,
            'sustainability_keyword_count': 0,
            'scrape_success': False,
            'scrape_timestamp': datetime.now().isoformat()
        }

        try:
            # Search for brand information using DuckDuckGo
            search_query = f"{brand_name}"
            if parent_company:
                search_query += f" {parent_company}"

            search_query += " brand company information"

            results = self._duckduckgo_search(search_query, max_results=3)

            for result in results:
                url = result.get('url', '')
                snippet = result.get('snippet', '')

                # Extract information from snippet
                if snippet:
                    brand_data['description'] = self._clean_text_for_csv(snippet)
                    brand_data['sustainability_keyword_count'] = self._count_keywords(snippet.lower())

                # Try to get more details from the page
                if url and 'wikipedia' in url.lower():
                    brand_data['website'] = url
                    page_info = self._scrape_page(url)
                    if page_info:
                        brand_data.update(page_info)
                        brand_data['scrape_success'] = True
                    break

            if self.verbose:
                if brand_data['scrape_success']:
                    print(f"  ✓ Brand info collected")
                else:
                    print(f"  ⚠ Limited brand info collected")

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Brand info scraping failed: {str(e)}")

        return brand_data

    def scrape_esg_data(self, brand_name: str, parent_company: Optional[str] = None) -> Dict:
        """
        Scrape ESG/CO2/Pollution data using DuckDuckGo and web scraping

        Args:
            brand_name: Brand name
            parent_company: Parent company name (optional)

        Returns:
            Dictionary with ESG data
        """
        esg_data = {
            'co2_emissions_mentioned': False,
            'pollution_mentioned': False,
            'esg_score_found': False,
            'sustainability_report_found': False,
            'carbon_neutral_claim': False,
            'esg_keyword_count': 0,
            'esg_text_snippets': [],
            'sources': [],
            'esg_data_found': False,
            'esg_scrape_timestamp': datetime.now().isoformat()
        }

        # Build search queries
        search_queries = [
            f"{brand_name} carbon footprint emissions",
            f"{brand_name} sustainability ESG report",
            f"{brand_name} environmental impact pollution"
        ]

        if parent_company:
            search_queries.append(f"{brand_name} {parent_company} carbon emissions")

        for query in search_queries:
            try:
                results = self._duckduckgo_search(query, max_results=3)

                for result in results:
                    url = result.get('url', '')
                    snippet = result.get('snippet', '')

                    if not snippet:
                        continue

                    # Clean snippet text
                    snippet_clean = self._clean_text_for_csv(snippet)
                    snippet_lower = snippet_clean.lower()

                    # Check for ESG indicators
                    if any(kw in snippet_lower for kw in ['co2', 'carbon', 'emissions', 'greenhouse']):
                        esg_data['co2_emissions_mentioned'] = True
                        if snippet_clean not in esg_data['esg_text_snippets']:
                            esg_data['esg_text_snippets'].append(snippet_clean)
                            esg_data['sources'].append(url)

                    if any(kw in snippet_lower for kw in ['pollution', 'pollutant', 'contamination']):
                        esg_data['pollution_mentioned'] = True
                        if snippet_clean not in esg_data['esg_text_snippets']:
                            esg_data['esg_text_snippets'].append(snippet_clean)
                            esg_data['sources'].append(url)

                    if any(kw in snippet_lower for kw in ['esg score', 'sustainability rating', 'environmental rating']):
                        esg_data['esg_score_found'] = True

                    if any(kw in snippet_lower for kw in ['sustainability report', 'esg report', 'environmental report']):
                        esg_data['sustainability_report_found'] = True

                    if any(kw in snippet_lower for kw in ['carbon neutral', 'net zero', 'carbon negative']):
                        esg_data['carbon_neutral_claim'] = True

                    # Count ESG keywords
                    for keyword in self.esg_keywords:
                        esg_data['esg_keyword_count'] += snippet_lower.count(keyword.lower())

                time.sleep(self.config['delay_between_requests'] / 2)  # Shorter delay between search queries

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ ESG search failed for query '{query}': {str(e)}")

        # Determine if any ESG data was found
        esg_data['esg_data_found'] = (
            esg_data['co2_emissions_mentioned'] or
            esg_data['pollution_mentioned'] or
            esg_data['esg_score_found'] or
            esg_data['sustainability_report_found'] or
            esg_data['esg_keyword_count'] > 0
        )

        if self.verbose:
            if esg_data['esg_data_found']:
                print(f"  ✓ ESG data collected:")
                print(f"    - CO2/Emissions: {'Yes' if esg_data['co2_emissions_mentioned'] else 'No'}")
                print(f"    - Pollution: {'Yes' if esg_data['pollution_mentioned'] else 'No'}")
                print(f"    - ESG keywords: {esg_data['esg_keyword_count']}")
                print(f"    - Sources: {len(esg_data['sources'])}")
            else:
                print(f"  ⚠ No ESG data found")

        # Convert lists to JSON strings for CSV export
        esg_data['esg_text_snippets'] = json.dumps(esg_data['esg_text_snippets'])
        esg_data['sources'] = json.dumps(esg_data['sources'])

        return esg_data

    def _duckduckgo_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform DuckDuckGo search using the official library"""
        try:
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                for result in search_results:
                    result_data = {
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', '')
                    }
                    results.append(result_data)
            return results
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ DuckDuckGo search error: {str(e)}")
            return []

    def _scrape_page(self, url: str) -> Optional[Dict]:
        """Scrape additional information from a webpage"""
        try:
            response = self.session.get(url, timeout=self.config['timeout'])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                info = {}

                # Try to extract from Wikipedia infobox
                infobox = soup.find('table', {'class': 'infobox'})
                if infobox:
                    sector = self._extract_infobox_field(infobox, ['industry', 'sector'])
                    category = self._extract_infobox_field(infobox, ['product type', 'products', 'type'])
                    info['sector'] = self._clean_text_for_csv(sector) if sector else None
                    info['category'] = self._clean_text_for_csv(category) if category else None

                # Extract first paragraph
                first_para = soup.find('p', class_=lambda x: x != 'mw-empty-elt')
                if first_para:
                    desc_text = first_para.get_text().strip()[:500]
                    info['description'] = self._clean_text_for_csv(desc_text)

                return info

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Page scraping failed: {str(e)}")
        return None

    def _extract_infobox_field(self, infobox, field_names: List[str]) -> Optional[str]:
        """Extract field from Wikipedia infobox"""
        for field_name in field_names:
            rows = infobox.find_all('tr')
            for row in rows:
                header = row.find('th')
                if header and field_name.lower() in header.get_text().lower():
                    value_cell = row.find('td')
                    if value_cell:
                        return value_cell.get_text().strip()
        return None

    def _count_keywords(self, text: str) -> int:
        """Count sustainability keywords in text"""
        if not text:
            return 0
        count = 0
        for keyword in self.keywords:
            count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text))
        return count

    @staticmethod
    def _clean_text_for_csv(text: str) -> str:
        """
        Clean text to prevent CSV formatting issues

        Args:
            text: Raw text string

        Returns:
            Cleaned text safe for CSV
        """
        if not text or pd.isna(text):
            return ""

        # Convert to string if not already
        text = str(text)

        # Replace newlines and carriage returns with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')

        # Replace tabs with spaces
        text = text.replace('\t', ' ')

        # Remove null bytes
        text = text.replace('\x00', '')

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Trim whitespace
        text = text.strip()

        # Remove or replace problematic characters
        # Remove control characters (except space)
        text = ''.join(char for char in text if ord(char) >= 32 or char == ' ')

        return text

    def _initialize_progress_file(self, csv_path: str):
        """Initialize progress file path"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        self.progress_file = f"{DATA_DIR}/{base_name}_esg_progress_{timestamp}.csv"

        if self.verbose:
            print(f"📝 Progress file initialized:")
            print(f"  {self.progress_file}\n")

    def _save_incremental_progress(self, original_df: pd.DataFrame,
                                   esg_data_list: List[Dict],
                                   brand_column: str):
        """Save progress incrementally"""
        if not self.progress_file:
            return

        try:
            # Create ESG DataFrame
            esg_df = pd.DataFrame(esg_data_list)

            # Merge with original
            result_df = original_df.merge(
                esg_df,
                left_on=brand_column,
                right_on='brand_name',
                how='left'
            )

            # Save to CSV
            result_df.to_csv(self.progress_file, index=False)

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Failed to save progress: {str(e)}")

    def export_to_csv(self, df: pd.DataFrame, output_path: str = None):
        """
        Export results to CSV

        Args:
            df: DataFrame to export
            output_path: Output file path (optional, auto-generated if not provided)
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{DATA_DIR}/brand_esg_results_{timestamp}.csv"

        df.to_csv(output_path, index=False)

        if self.verbose:
            print(f"📁 Results exported to: {output_path}")
            print(f"   Total rows: {len(df)}")
            print(f"   Total columns: {len(df.columns)}")

        return output_path


def main():
    """Example usage"""
    import sys

    print("="*80)
    print("CSV Brand Scraper - Example Usage")
    print("="*80)

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        print("\nUsage: python csv_brand_scraper.py <path_to_csv>")
        print("\nExample CSV format:")
        print("  brand_name,parent_company")
        print("  Pepsi,PepsiCo")
        print("  Lay's,PepsiCo")
        print("  Coca-Cola,The Coca-Cola Company")
        print("\nRequired columns:")
        print("  - brand_name (or specify with --brand-column)")
        print("  - parent_company (optional)")
        return

    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"\n❌ Error: File not found: {csv_path}")
        return

    # Parse optional arguments
    save_progress = '--save-progress' in sys.argv or '-p' in sys.argv
    brand_column = 'brand_name'
    company_column = 'parent_company'

    # Initialize scraper
    scraper = CSVBrandScraper(verbose=True, save_progress=save_progress)

    # Scrape data
    result_df = scraper.scrape_from_csv(
        csv_path,
        brand_column=brand_column,
        company_column=company_column
    )

    # Export results
    if not result_df.empty:
        output_path = scraper.export_to_csv(result_df)

        print("\n" + "="*80)
        print("✅ Complete!")
        print("="*80)
        print(f"\nOutput file: {output_path}")
        print(f"Total brands: {len(result_df)}")

        # Show summary statistics
        if 'esg_data_found' in result_df.columns:
            print(f"\n📊 ESG Data Summary:")
            print(f"  Brands with ESG data: {result_df['esg_data_found'].sum()}")
            if 'co2_emissions_mentioned' in result_df.columns:
                print(f"  CO2/Emissions mentions: {result_df['co2_emissions_mentioned'].sum()}")
            if 'pollution_mentioned' in result_df.columns:
                print(f"  Pollution mentions: {result_df['pollution_mentioned'].sum()}")
            if 'carbon_neutral_claim' in result_df.columns:
                print(f"  Carbon neutral claims: {result_df['carbon_neutral_claim'].sum()}")

        print("="*80)


if __name__ == "__main__":
    main()
