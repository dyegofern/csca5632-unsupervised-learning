"""
Enhanced Web Scraper Module
Scrapes company brands and ESG/CO2/Pollution data using DuckDuckGo
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from typing import Optional, Dict, List, Tuple
from urllib.parse import quote_plus, urljoin
import warnings
import json
from datetime import datetime

# Suppress all warnings
warnings.filterwarnings('ignore')

try:
    from ddgs import DDGS
except ImportError:
    # Fallback to old package name if ddgs not installed
    from duckduckgo_search import DDGS

from config import SCRAPING_CONFIG, SUSTAINABILITY_KEYWORDS, VERBOSE, DATA_DIR


class EnhancedCompanyScraper:
    """
    Enhanced scraper that:
    1. Discovers brands owned by a parent company using DuckDuckGo
    2. Collects ESG/CO2/Pollution information for each brand
    3. Exports data in CSV format compatible with existing database
    """

    def __init__(self, config: dict = SCRAPING_CONFIG,
                 keywords: list = SUSTAINABILITY_KEYWORDS,
                 verbose: bool = VERBOSE,
                 save_progress: bool = False):
        """
        Initialize enhanced company scraper

        Args:
            config: Scraping configuration (delays, retries, etc.)
            keywords: List of sustainability-related keywords to search for
            verbose: Print progress messages
            save_progress: Save data incrementally after each brand (default: False)
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

        # ESG-specific keywords for more targeted searches
        self.esg_keywords = [
            'carbon footprint', 'co2 emissions', 'greenhouse gas',
            'pollution', 'environmental impact', 'sustainability report',
            'esg score', 'carbon neutral', 'net zero', 'renewable energy',
            'water usage', 'waste reduction', 'circular economy',
            'environmental compliance', 'climate change'
        ]

        # For incremental saving
        self.progress_file_brands = None
        self.progress_file_esg = None
        self.progress_file_combined = None

    def scrape_company_brands_and_esg(self, company_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method: Discover brands and collect ESG data for a company

        Args:
            company_name: Name of the parent company (e.g., "PepsiCo", "Coca-Cola")

        Returns:
            Tuple of (brands_df, esg_df):
                - brands_df: DataFrame with brand information
                - esg_df: DataFrame with ESG/CO2/Pollution data
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Starting enhanced scraping for: {company_name}")
            if self.save_progress:
                print(f"💾 Incremental saving: ENABLED")
            print(f"{'='*80}\n")

        # Step 1: Discover brands owned by the company
        brands = self.discover_company_brands(company_name)

        if not brands:
            if self.verbose:
                print(f"⚠ No brands discovered for {company_name}")
            return pd.DataFrame(), pd.DataFrame()

        # Initialize progress files if incremental saving is enabled
        if self.save_progress:
            self._initialize_progress_files(company_name)

        # Step 2: Collect ESG data for each brand
        brands_data = []
        esg_data = []

        for i, brand_name in enumerate(brands, 1):
            if self.verbose:
                print(f"\n[{i}/{len(brands)}] Processing: {brand_name}")
                print("-" * 60)

            # Get brand information
            brand_info = self.scrape_brand_info(brand_name, company_name)
            brands_data.append(brand_info)

            # Get ESG information
            esg_info = self.scrape_esg_data(brand_name, company_name)
            esg_data.append(esg_info)

            # Save progress after each brand if enabled
            if self.save_progress:
                self._save_incremental_progress(brands_data, esg_data, company_name)
                if self.verbose:
                    print(f"  💾 Progress saved ({i}/{len(brands)} brands)")

            time.sleep(self.config['delay_between_requests'])

        # Create DataFrames
        brands_df = pd.DataFrame(brands_data)
        esg_df = pd.DataFrame(esg_data)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✓ Scraping complete for {company_name}")
            print(f"  Brands found: {len(brands)}")
            print(f"  Brands with data: {brands_df['scrape_success'].sum()}")
            print(f"  ESG data collected: {esg_df['esg_data_found'].sum()}")
            print(f"{'='*80}\n")

        return brands_df, esg_df

    def discover_company_brands(self, company_name: str) -> List[str]:
        """
        Discover brands owned by a company using DuckDuckGo search

        Args:
            company_name: Parent company name

        Returns:
            List of brand names
        """
        if self.verbose:
            print(f"🔍 Discovering brands owned by {company_name}...")

        brands = set()

        # First try Wikipedia (most reliable)
        wiki_brands = self._scrape_wikipedia_brands(company_name)
        brands.update(wiki_brands)

        if self.verbose:
            print(f"  → Found {len(wiki_brands)} brands from Wikipedia")

        # Then try DuckDuckGo searches
        search_queries = [
            f"{company_name} brands list",
            f"{company_name} portfolio of brands",
            f"brands owned by {company_name}",
        ]

        for query in search_queries:
            try:
                results = self._duckduckgo_search(query, max_results=5)
                discovered = self._extract_brands_from_results(results, company_name)
                brands.update(discovered)
                if self.verbose:
                    print(f"  → Found {len(discovered)} new brands from: '{query}'")
                time.sleep(self.config['delay_between_requests'])
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Search failed for query '{query}': {str(e)}")

        brands_list = sorted(list(brands))

        if self.verbose:
            print(f"  ✓ Discovered {len(brands_list)} total brands")
            if brands_list:
                print(f"    Brands: {', '.join(brands_list[:10])}")
                if len(brands_list) > 10:
                    print(f"    ... and {len(brands_list) - 10} more")

        return brands_list

    def _duckduckgo_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Perform DuckDuckGo search using the official library

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search result dictionaries
        """
        try:
            results = []

            # Use DDGS library for text search
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

    def _extract_brands_from_results(self, search_results: List[Dict],
                                     company_name: str) -> List[str]:
        """
        Extract brand names from search results

        Args:
            search_results: List of search result dictionaries
            company_name: Parent company name for context

        Returns:
            List of brand names
        """
        brands = set()

        # First, extract from snippets (faster, no extra requests)
        for result in search_results:
            snippet = result.get('snippet', '')
            if snippet:
                # Extract capitalized words that could be brand names
                potential_brands = re.findall(r'\b[A-Z][A-Za-z\s&\'-]{1,28}\b', snippet)
                brands.update([self._clean_text_for_csv(b.strip()) for b in potential_brands if len(b.strip()) > 2])

        # Then fetch full pages for more detailed extraction (only for promising URLs)
        for result in search_results:
            url = result.get('url', '')
            title = result.get('title', '').lower()

            # Only fetch if URL seems relevant (contains keywords like 'brand', 'portfolio', etc.)
            if url and any(keyword in title for keyword in ['brand', 'portfolio', 'product', 'subsidiary']):
                try:
                    response = self._make_request(url)
                    if response and response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')

                        # Look for lists of brands (common on brand portfolio pages)
                        lists = soup.find_all(['ul', 'ol'], limit=20)  # Limit to first 20 lists
                        for lst in lists:
                            items = lst.find_all('li', limit=50)  # Limit items per list
                            for item in items:
                                text = item.get_text().strip()
                                # Extract potential brand names
                                potential_brands = re.findall(r'\b[A-Z][A-Za-z\s&\'-]{1,28}\b', text)
                                brands.update([self._clean_text_for_csv(b.strip()) for b in potential_brands if len(b.strip()) > 2])

                        time.sleep(self.config['delay_between_requests'])

                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Failed to extract from {url[:50]}...")
                    continue

        # Filter out common non-brand words and the company name itself
        filtered_brands = self._filter_brand_names(brands, company_name)

        return filtered_brands

    def _filter_brand_names(self, brands: set, company_name: str) -> List[str]:
        """
        Filter out non-brand names from discovered brands

        Args:
            brands: Set of potential brand names
            company_name: Parent company name

        Returns:
            Filtered list of brand names
        """
        # Common words to exclude (Wikipedia navigation, generic terms, etc.)
        exclude_words = {
            # Generic business terms
            'Company', 'Corporation', 'Inc', 'Ltd', 'LLC', 'The', 'And', 'Or',
            'Products', 'Services', 'Brand', 'Brands', 'Portfolio',
            'Contact', 'Home', 'News', 'Press', 'Media', 'Careers', 'Privacy',
            'Terms', 'Conditions', 'Policy', 'Copyright', 'All', 'Rights',
            'Reserved', company_name, 'Menu', 'Search', 'Login', 'Register',

            # Wikipedia navigation elements
            'About Wikipedia', 'Article', 'Asset lists', 'Category', 'Commons',
            'Community portal', 'Contact Wikipedia', 'Contact us', 'Contents',
            'Cookie statement', 'Create account', 'Current events', 'Developers',
            'Disclaimers', 'Donate', 'Download QR code', 'Download as PDF', 'Edit',
            'Get shortened URL', 'Help', 'Learn to edit', 'Log in', 'Main page',
            'Mobile view', 'Next', 'Page information', 'Permanent link', 'Printable version',
            'Privacy policy', 'Random article', 'Read', 'Recent changes', 'Related changes',
            'Special pages', 'Statistics', 'Talk', 'Upload file', 'View history',
            'What links here', 'Wikidata item',

            # Generic food terms
            'Diet', 'Caffeine Free', 'Zero Sugar', 'Natural', 'Baked!', 'Crunchy',

            # People names (add known executives)
            'Indra Nooyi', 'Ramon Laguarta', 'Donald M. Kendall', 'Steven Reinemund',
            'Robert E. Allen', 'Dina Dublon', 'Caleb Bradham',

            # Events/concepts
            'Cola wars', 'Leonard v. Pepsico', 'Pepsi Number Fever', 'Pepsi Stuff'
        }

        # Patterns to exclude
        exclude_patterns = [
            'http', 'www', '.com', 'click here', 'read more',
            'subsidiaries', 'brands owned', 'list of'
        ]

        filtered = []
        for brand in brands:
            brand_clean = brand.strip()

            # Skip if too short or too long
            if len(brand_clean) < 3 or len(brand_clean) > 40:
                continue

            # Skip if in exclude list (case-insensitive)
            if brand_clean in exclude_words or brand_clean.lower() in [e.lower() for e in exclude_words]:
                continue

            # Skip if mostly lowercase (likely not a proper noun/brand)
            if brand_clean.islower():
                continue

            # Skip if contains common non-brand patterns
            if any(pattern in brand_clean.lower() for pattern in exclude_patterns):
                continue

            # Skip if it's a single common word
            if brand_clean in ['Diet', 'Life', 'Max', 'Next', 'Star', 'Jazz', 'Read', 'Edit', 'Help', 'Rice']:
                continue

            # Skip if it looks like a flavor descriptor without brand
            if brand_clean.endswith(('Red', 'Blue', 'Green', 'Gold', 'Pink', 'White', 'Black')) and len(brand_clean.split()) == 1:
                continue

            filtered.append(brand_clean)

        return filtered

    def _scrape_wikipedia_brands(self, company_name: str) -> List[str]:
        """
        Scrape brand information from company's Wikipedia page

        Args:
            company_name: Parent company name

        Returns:
            List of brand names
        """
        brands = []

        try:
            # Try main company page first
            wiki_url = f"https://en.wikipedia.org/wiki/{quote_plus(company_name)}"
            response = self._make_request(wiki_url)

            if response and response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for sections about brands/products/subsidiaries
                headings = soup.find_all(['h2', 'h3', 'h4'])

                for heading in headings:
                    heading_text = heading.get_text().lower()
                    if any(keyword in heading_text for keyword in ['brand', 'product', 'subsidiary', 'portfolio', 'division']):
                        # Get content after this heading
                        next_elem = heading.find_next_sibling()
                        while next_elem and next_elem.name not in ['h2', 'h3']:
                            if next_elem.name in ['ul', 'ol']:
                                items = next_elem.find_all('li', limit=100)
                                for item in items:
                                    text = item.get_text().strip()
                                    # Extract brand name (before any parenthesis, dash, or newline)
                                    # More flexible pattern to catch various formats
                                    text_clean = text.split('(')[0].split('[')[0].split('–')[0].split('-')[0].strip()
                                    brand_match = re.match(r'^([A-Z][A-Za-z0-9\s&\'.!-]{1,35})', text_clean)
                                    if brand_match:
                                        brand_name = self._clean_text_for_csv(brand_match.group(1).strip())
                                        if len(brand_name) > 2:
                                            brands.append(brand_name)
                            next_elem = next_elem.find_next_sibling()

            # Also try "List of [Company] brands" page if it exists
            list_url = f"https://en.wikipedia.org/wiki/List_of_{quote_plus(company_name)}_brands"
            response2 = self._make_request(list_url)

            if response2 and response2.status_code == 200:
                soup2 = BeautifulSoup(response2.content, 'html.parser')

                # Only look at content, not navigation/footer
                content_div = soup2.find('div', {'id': 'mw-content-text'})
                if content_div:
                    # Find all lists within the main content
                    lists = content_div.find_all(['ul', 'ol'])
                    for lst in lists:
                        # Skip if list is in navigation/reference sections
                        parent_classes = lst.parent.get('class', []) if lst.parent else []
                        if any(cls in str(parent_classes) for cls in ['navbox', 'reflist', 'navbox', 'sidebar']):
                            continue

                        items = lst.find_all('li', limit=200)
                        for item in items:
                            text = item.get_text().strip()

                            # Skip if it looks like a reference or footnote
                            if text.startswith('^') or re.match(r'^\d+\.?\d*\s', text):
                                continue

                            text_clean = text.split('(')[0].split('[')[0].split('–')[0].split('-')[0].strip()
                            brand_match = re.match(r'^([A-Z][A-Za-z0-9\s&\'.!-]{1,35})', text_clean)
                            if brand_match:
                                brand_name = self._clean_text_for_csv(brand_match.group(1).strip())
                                if len(brand_name) > 2:
                                    brands.append(brand_name)

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Wikipedia scraping failed: {str(e)}")

        # Remove duplicates and filter
        brands_unique = list(set(brands))
        filtered = self._filter_brand_names(set(brands_unique), company_name)

        return filtered

    def scrape_brand_info(self, brand_name: str, parent_company: str) -> Dict:
        """
        Scrape general brand information

        Args:
            brand_name: Brand name
            parent_company: Parent company name

        Returns:
            Dictionary with brand information
        """
        brand_data = {
            'brand_name': brand_name,
            'parent_company': parent_company,
            'sector': None,
            'category': None,
            'description': None,
            'wikipedia_url': None,
            'sustainability_keyword_count': 0,
            'scrape_success': False,
            'scrape_timestamp': datetime.now().isoformat()
        }

        try:
            # Try Wikipedia
            wiki_url = f"https://en.wikipedia.org/wiki/{quote_plus(brand_name)}"
            response = self._make_request(wiki_url)

            if response and response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract infobox data
                infobox = soup.find('table', {'class': 'infobox'})
                if infobox:
                    sector = self._extract_infobox_field(infobox, ['industry', 'sector'])
                    category = self._extract_infobox_field(infobox, ['product type', 'products', 'type'])
                    brand_data['sector'] = self._clean_text_for_csv(sector) if sector else None
                    brand_data['category'] = self._clean_text_for_csv(category) if category else None

                # Extract description
                first_para = soup.find('p', class_=lambda x: x != 'mw-empty-elt')
                if first_para:
                    description = first_para.get_text().strip()
                    brand_data['description'] = self._clean_text_for_csv(description)
                    brand_data['sustainability_keyword_count'] = self._count_keywords(description.lower())

                brand_data['wikipedia_url'] = wiki_url
                brand_data['scrape_success'] = True

                if self.verbose:
                    print(f"  ✓ Brand info collected")

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Brand info scraping failed: {str(e)}")

        return brand_data

    def scrape_esg_data(self, brand_name: str, parent_company: str) -> Dict:
        """
        Scrape ESG/CO2/Pollution data for a brand

        Args:
            brand_name: Brand name
            parent_company: Parent company name

        Returns:
            Dictionary with ESG data
        """
        esg_data = {
            'brand_name': brand_name,
            'parent_company': parent_company,
            'co2_emissions_mentioned': False,
            'pollution_mentioned': False,
            'esg_score_found': False,
            'sustainability_report_found': False,
            'carbon_neutral_claim': False,
            'esg_keyword_count': 0,
            'esg_text_snippets': [],
            'sources': [],
            'esg_data_found': False,
            'scrape_timestamp': datetime.now().isoformat()
        }

        # Search queries for ESG data
        search_queries = [
            f"{brand_name} {parent_company} carbon footprint",
            f"{brand_name} ESG sustainability report",
            f"{brand_name} environmental impact CO2",
            f"{brand_name} pollution emissions"
        ]

        for query in search_queries:
            try:
                results = self._duckduckgo_search(query, max_results=3)

                for result in results:
                    url = result.get('url', '')
                    snippet = result.get('snippet', '')

                    # Clean snippet text
                    snippet_clean = self._clean_text_for_csv(snippet)
                    snippet_lower = snippet_clean.lower()

                    if any(kw in snippet_lower for kw in ['co2', 'carbon', 'emissions']):
                        esg_data['co2_emissions_mentioned'] = True
                        esg_data['esg_text_snippets'].append(snippet_clean)
                        esg_data['sources'].append(url)

                    if any(kw in snippet_lower for kw in ['pollution', 'pollutant', 'contamination']):
                        esg_data['pollution_mentioned'] = True
                        if snippet_clean not in esg_data['esg_text_snippets']:
                            esg_data['esg_text_snippets'].append(snippet_clean)
                            esg_data['sources'].append(url)

                    if any(kw in snippet_lower for kw in ['esg score', 'sustainability rating']):
                        esg_data['esg_score_found'] = True

                    if any(kw in snippet_lower for kw in ['sustainability report', 'esg report']):
                        esg_data['sustainability_report_found'] = True

                    if any(kw in snippet_lower for kw in ['carbon neutral', 'net zero', 'carbon negative']):
                        esg_data['carbon_neutral_claim'] = True

                    # Count ESG keywords
                    for keyword in self.esg_keywords:
                        esg_data['esg_keyword_count'] += snippet_lower.count(keyword.lower())

                time.sleep(self.config['delay_between_requests'])

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

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.config['max_retries']):
            try:
                response = self.session.get(url, timeout=self.config['timeout'])
                return response
            except requests.RequestException as e:
                if attempt == self.config['max_retries'] - 1:
                    if self.verbose:
                        print(f"  ⚠ Request failed: {url[:50]}...")
                    return None
                time.sleep(self.config['delay_between_requests'])
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

        # Escape quotes by doubling them (CSV standard)
        # Pandas will handle this automatically, but being explicit doesn't hurt

        return text

    def _initialize_progress_files(self, company_name: str):
        """
        Initialize progress file paths for incremental saving

        Args:
            company_name: Company name for filename
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        company_clean = company_name.replace(' ', '_').replace('.', '')

        self.progress_file_brands = f"{DATA_DIR}/{company_clean}_brands_progress_{timestamp}.csv"
        self.progress_file_esg = f"{DATA_DIR}/{company_clean}_esg_progress_{timestamp}.csv"
        self.progress_file_combined = f"{DATA_DIR}/{company_clean}_complete_progress_{timestamp}.csv"

        if self.verbose:
            print(f"📝 Progress files initialized:")
            print(f"  - Brands: {self.progress_file_brands}")
            print(f"  - ESG: {self.progress_file_esg}")
            print(f"  - Combined: {self.progress_file_combined}")

    def _save_incremental_progress(self, brands_data: List[Dict],
                                   esg_data: List[Dict],
                                   company_name: str):
        """
        Save progress incrementally after each brand

        Args:
            brands_data: List of brand data dictionaries
            esg_data: List of ESG data dictionaries
            company_name: Company name
        """
        if not self.progress_file_brands or not self.progress_file_esg:
            return

        try:
            # Convert to DataFrames
            brands_df = pd.DataFrame(brands_data)
            esg_df = pd.DataFrame(esg_data)

            # Save individual files
            brands_df.to_csv(self.progress_file_brands, index=False)
            esg_df.to_csv(self.progress_file_esg, index=False)

            # Create and save combined file
            combined_df = brands_df.merge(
                esg_df,
                on=['brand_name', 'parent_company'],
                how='left',
                suffixes=('', '_esg')
            )

            # Select and rename columns for database compatibility
            final_columns = {
                'brand_name': 'brand_name',
                'parent_company': 'parent_company',
                'sector': 'sector',
                'category': 'category',
                'description': 'description',
                'sustainability_keyword_count': 'sustainability_keywords',
                'co2_emissions_mentioned': 'has_co2_data',
                'pollution_mentioned': 'has_pollution_data',
                'esg_score_found': 'has_esg_score',
                'sustainability_report_found': 'has_sustainability_report',
                'carbon_neutral_claim': 'claims_carbon_neutral',
                'esg_keyword_count': 'esg_keyword_count',
                'esg_text_snippets': 'esg_snippets',
                'sources': 'data_sources',
                'wikipedia_url': 'wikipedia_url',
                'scrape_success': 'data_quality_flag'
            }

            combined_df = combined_df.rename(columns=final_columns)
            combined_df = combined_df[[col for col in final_columns.values() if col in combined_df.columns]]

            combined_df.to_csv(self.progress_file_combined, index=False)

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Failed to save progress: {str(e)}")

    def export_to_csv(self, brands_df: pd.DataFrame, esg_df: pd.DataFrame,
                     company_name: str, output_dir: str = DATA_DIR):
        """
        Export scraped data to CSV files

        Args:
            brands_df: DataFrame with brand information
            esg_df: DataFrame with ESG data
            company_name: Company name for filename
            output_dir: Directory to save CSV files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        company_clean = company_name.replace(' ', '_').replace('.', '')

        brands_file = f"{output_dir}/{company_clean}_brands_{timestamp}.csv"
        esg_file = f"{output_dir}/{company_clean}_esg_{timestamp}.csv"

        # Export brands
        brands_df.to_csv(brands_file, index=False)

        # Export ESG data
        esg_df.to_csv(esg_file, index=False)

        if self.verbose:
            print(f"\n📁 Data exported:")
            print(f"  - Brands: {brands_file}")
            print(f"  - ESG Data: {esg_file}")

        return brands_file, esg_file

    def create_combined_export(self, brands_df: pd.DataFrame, esg_df: pd.DataFrame,
                              company_name: str, output_dir: str = DATA_DIR):
        """
        Create a combined CSV with all data for database import

        Args:
            brands_df: DataFrame with brand information
            esg_df: DataFrame with ESG data
            company_name: Company name
            output_dir: Directory to save CSV file

        Returns:
            Path to combined CSV file
        """
        # Merge brands and ESG data
        combined_df = brands_df.merge(
            esg_df,
            on=['brand_name', 'parent_company'],
            how='left',
            suffixes=('', '_esg')
        )

        # Select and rename columns for database compatibility
        final_columns = {
            'brand_name': 'brand_name',
            'parent_company': 'parent_company',
            'sector': 'sector',
            'category': 'category',
            'description': 'description',
            'sustainability_keyword_count': 'sustainability_keywords',
            'co2_emissions_mentioned': 'has_co2_data',
            'pollution_mentioned': 'has_pollution_data',
            'esg_score_found': 'has_esg_score',
            'sustainability_report_found': 'has_sustainability_report',
            'carbon_neutral_claim': 'claims_carbon_neutral',
            'esg_keyword_count': 'esg_keyword_count',
            'esg_text_snippets': 'esg_snippets',
            'sources': 'data_sources',
            'wikipedia_url': 'wikipedia_url',
            'scrape_success': 'data_quality_flag'
        }

        combined_df = combined_df.rename(columns=final_columns)
        combined_df = combined_df[[col for col in final_columns.values() if col in combined_df.columns]]

        # Export combined file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        company_clean = company_name.replace(' ', '_').replace('.', '')
        combined_file = f"{output_dir}/{company_clean}_complete_{timestamp}.csv"

        combined_df.to_csv(combined_file, index=False)

        if self.verbose:
            print(f"\n📊 Combined export created: {combined_file}")
            print(f"   Total records: {len(combined_df)}")
            print(f"   Columns: {len(combined_df.columns)}")

        return combined_file


def main():
    """Example usage of the enhanced scraper"""
    print("="*80)
    print("Enhanced Company Brand & ESG Scraper")
    print("="*80)

    # Initialize scraper
    scraper = EnhancedCompanyScraper(verbose=True)

    # Example: Scrape PepsiCo
    company_name = "PepsiCo"
    print(f"\nScraping data for: {company_name}")

    # Perform scraping
    brands_df, esg_df = scraper.scrape_company_brands_and_esg(company_name)

    # Export data
    if not brands_df.empty:
        scraper.export_to_csv(brands_df, esg_df, company_name)
        combined_file = scraper.create_combined_export(brands_df, esg_df, company_name)

        print("\n" + "="*80)
        print("Summary Statistics:")
        print("="*80)
        print(f"Total brands discovered: {len(brands_df)}")
        print(f"Brands with valid data: {brands_df['scrape_success'].sum()}")
        print(f"Brands with ESG data: {esg_df['esg_data_found'].sum()}")
        print(f"\nESG Metrics:")
        print(f"  - CO2/Emissions mentions: {esg_df['co2_emissions_mentioned'].sum()}")
        print(f"  - Pollution mentions: {esg_df['pollution_mentioned'].sum()}")
        print(f"  - Carbon neutral claims: {esg_df['carbon_neutral_claim'].sum()}")
        print(f"  - Sustainability reports found: {esg_df['sustainability_report_found'].sum()}")
        print("="*80)
    else:
        print("\n⚠ No data collected. Please check the company name and try again.")


if __name__ == "__main__":
    main()
