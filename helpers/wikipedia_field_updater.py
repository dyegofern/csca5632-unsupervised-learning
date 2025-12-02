"""
Wikipedia Field Updater Script
Updates specific fields in a CSV file using Wikipedia as the sole data source
Author: AI Assistant
Date: 2025-11-22
"""

import pandas as pd
import time
import re
import requests
from typing import Dict, Optional, List
import logging
from datetime import datetime
import argparse
import sys
import unicodedata

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wikipedia_field_updater.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def sanitize_for_console(text: str) -> str:
    """Remove non-ASCII characters to prevent encoding errors on Windows console"""
    if not isinstance(text, str):
        text = str(text)
    # Replace common Unicode characters and then remove any remaining non-ASCII
    replacements = {
        '\u2192': '->',
        '\u2713': '+',
        '\u2717': 'X',
        '\u2298': '-',
        '\ufffd': '?',
        '\u2014': '--',
        '\u2013': '-',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    text = text.encode('ascii', errors='replace').decode('ascii')
    return text

class WikipediaFieldUpdater:
    """Update specific CSV fields using Wikipedia data"""

    # Fields to update (as specified by user)
    FIELDS_TO_UPDATE = [
        'country_of_origin',
        'year_of_foundation',
        'headquarters_country',
        'employees',
        'fossil_fuel_reliance',
        'esg_programs',
        'sustainability_actions',
        'target_population',
        'customer_loyalty_index',
        'r_and_d_spend_percent_revenue',
        'women_board_percent',
        'ceo_tenure_years',
        'market_cap_billion_usd',
        'references_and_links'
    ]

    def __init__(self, input_csv: str, output_csv: str, brand_column: str = 'brand_name',
                 verbose: bool = True, rate_limit: float = 1.0):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.brand_column = brand_column
        self.verbose = verbose
        self.rate_limit_delay = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikipediaFieldUpdater/1.0 (Educational/Research Purpose)'
        })
        # Note: DDGS instance will be created fresh for each search to avoid blocking

        logging.info("="*80)
        logging.info("Wikipedia Field Updater Initialized")
        logging.info("="*80)
        logging.info(f"Input CSV: {input_csv}")
        logging.info(f"Output CSV: {output_csv}")
        logging.info(f"Brand column: {brand_column}")
        logging.info(f"Rate limit: {rate_limit}s between requests")
        logging.info(f"Fields to update: {', '.join(self.FIELDS_TO_UPDATE)}")

    def generate_search_variations(self, brand_name: str) -> List[str]:
        """
        Generate variations of brand name for better search results
        Handles: number/word conversion, special characters, qualifiers, etc.
        """
        variations = [brand_name]  # Always include original

        # Number to word mapping
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve'
        }

        # Word to number mapping (reverse)
        word_numbers = {v: k for k, v in number_words.items()}

        # Try converting numbers to words
        name_lower = brand_name.lower()
        for num, word in number_words.items():
            if num in brand_name:
                variations.append(brand_name.replace(num, word))
                variations.append(brand_name.replace(num, word.capitalize()))

        # Try converting words to numbers
        for word, num in word_numbers.items():
            if word in name_lower:
                # Replace whole word only
                pattern = r'\b' + word + r'\b'
                if re.search(pattern, name_lower, re.IGNORECASE):
                    variations.append(re.sub(pattern, num, brand_name, flags=re.IGNORECASE))

        # Add common brand qualifiers (for disambiguation)
        # These help find pages like "3 Musketeers (chocolate bar)"
        brand_qualifiers = ['brand', 'company', 'product']
        for qualifier in brand_qualifiers:
            variations.append(f"{brand_name} {qualifier}")

        # Remove special characters version
        cleaned = re.sub(r'[^\w\s]', '', brand_name)
        if cleaned != brand_name:
            variations.append(cleaned)

        # Remove accents/diacritics
        normalized = unicodedata.normalize('NFKD', brand_name)
        ascii_version = normalized.encode('ASCII', 'ignore').decode('ASCII')
        if ascii_version != brand_name and ascii_version.strip():
            variations.append(ascii_version)

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            v_stripped = v.strip()
            if v_stripped and v_stripped not in seen:
                seen.add(v_stripped)
                unique_variations.append(v_stripped)

        # Limit to original + 6 variations (total of 7 max)
        # Original is always first, so keep it + next 6
        if len(unique_variations) > 7:
            unique_variations = unique_variations[:7]

        return unique_variations

    def get_wikipedia_page(self, brand_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Get Wikipedia page content and URL for a brand
        Tries multiple language editions in order: en, pt, es, fr, de, ja, zh
        Returns: (page_content, page_title, page_url)
        """
        # Languages to try in order: English, Portuguese, Spanish, French, German, Japanese, Chinese
        languages = [
            ('en', 'English'),
            ('pt', 'Portuguese'),
            ('es', 'Spanish'),
            ('fr', 'French'),
            ('de', 'German'),
            ('ja', 'Japanese'),
            ('zh', 'Chinese')
        ]

        # Generate search variations (but keep original first)
        search_variations = self.generate_search_variations(brand_name)

        # Ensure original is first, then add other variations
        # (generate_search_variations already puts original first, but let's be explicit)
        if search_variations[0] != brand_name:
            search_variations = [brand_name] + [v for v in search_variations if v != brand_name]

        variations_to_try = [brand_name]  # Only original for first pass
        other_variations = [v for v in search_variations if v != brand_name]

        if self.verbose:
            logging.info(f"  > Searching Wikipedia for: '{sanitize_for_console(brand_name)}'")
            if other_variations:
                logging.info(f"  > Generated {len(other_variations)} alternative search variations for fallback")
            logging.info(f"  > Will try languages in order: {', '.join([name for _, name in languages])}")

        # PASS 1: Try original name across all languages first
        for lang_code, lang_name in languages:
            for search_term in [brand_name]:
                try:
                    if self.verbose and search_term == search_variations[0]:
                        logging.info(f"    -> Trying {lang_name} Wikipedia ({lang_code}.wikipedia.org)...")
                    elif self.verbose and search_term != search_variations[0]:
                        logging.info(f"       Trying variation: '{sanitize_for_console(search_term)}'")

                    time.sleep(self.rate_limit_delay)

                    # Use Wikipedia API to search
                    search_url = f"https://{lang_code}.wikipedia.org/w/api.php"
                    search_params = {
                        'action': 'opensearch',
                        'search': search_term,
                        'limit': 5,  # Get more results to choose from
                        'format': 'json'
                    }

                    response = self.session.get(search_url, params=search_params, timeout=10)

                    if response.status_code == 200:
                        results = response.json()
                        if len(results) > 1 and len(results[1]) > 0:
                            # results[1] contains titles, results[3] contains URLs
                            titles = results[1]
                            urls = results[3] if len(results) > 3 else []

                            # Try to find the best match
                            # Priority: exact match > brand-related > any match
                            best_match_idx = None

                            # Brand-related keywords to prioritize
                            brand_keywords = ['brand', 'company', 'product', 'chocolate', 'candy',
                                            'food', 'beverage', 'corporation', 'manufacturer',
                                            'soft drink', 'soda', 'bar', 'snack', 'retailer',
                                            'chain', 'store']

                            # Keywords that indicate NON-brand pages (to exclude)
                            exclude_keywords = ['film', 'movie', 'television', 'tv series', 'novel',
                                              'book', 'album', 'song', 'band', 'actor', 'actress',
                                              'director', 'singer', 'musician', 'character',
                                              'video game', 'game', 'play', 'opera', 'anime',
                                              'manga', 'comic', 'series', 'episode', 'season']

                            # First, look for brand-related pages
                            for idx, title in enumerate(titles):
                                title_lower = title.lower()

                                # Skip if this looks like a non-brand page
                                if any(exclude_word in title_lower for exclude_word in exclude_keywords):
                                    if self.verbose:
                                        logging.info(f"       Skipping non-brand result: '{sanitize_for_console(title)}'")
                                    continue

                                # Check if title contains brand name (exact or close match)
                                brand_name_clean = re.sub(r'[^\w\s]', '', brand_name).lower()
                                title_clean = re.sub(r'[^\w\s]', '', title).lower()

                                # Exact match or match with qualifier (like "3 Musketeers (chocolate bar)")
                                if brand_name_clean in title_clean:
                                    # Check if it's a brand/product page
                                    if any(keyword in title_lower for keyword in brand_keywords):
                                        best_match_idx = idx
                                        if self.verbose:
                                            logging.info(f"       Found brand-related match: '{sanitize_for_console(title)}'")
                                        break
                                    # If no brand keyword but name matches, still use it (but keep looking)
                                    elif best_match_idx is None:
                                        best_match_idx = idx

                            # If no specific match found, try to find first non-excluded result
                            if best_match_idx is None:
                                for idx, title in enumerate(titles):
                                    title_lower = title.lower()
                                    if not any(exclude_word in title_lower for exclude_word in exclude_keywords):
                                        best_match_idx = idx
                                        break

                            # If still no match (all results were excluded), skip this search term
                            if best_match_idx is None:
                                if self.verbose:
                                    logging.info(f"       All results filtered out as non-brand pages")
                                continue  # Try next language

                            page_title = titles[best_match_idx]
                            page_url = urls[best_match_idx] if best_match_idx < len(urls) else None

                            if self.verbose and best_match_idx != 0:
                                logging.info(f"       Selected result #{best_match_idx + 1} as best match")
                            if self.verbose:
                                logging.info(f"       Found match: '{sanitize_for_console(page_title)}'")
                                if search_term != brand_name:
                                    logging.info(f"       (matched using variation: '{sanitize_for_console(search_term)}')")
                                logging.info(f"       Fetching page content...")

                            # Get page content
                            content_params = {
                                'action': 'query',
                                'titles': page_title,
                                'prop': 'extracts|revisions',
                                'explaintext': True,
                                'rvprop': 'content',
                                'format': 'json'
                            }

                            content_response = self.session.get(search_url, params=content_params, timeout=10)

                            if content_response.status_code == 200:
                                data = content_response.json()
                                pages = data.get('query', {}).get('pages', {})

                                for page_id, page_data in pages.items():
                                    extract = page_data.get('extract', '')
                                    if extract:
                                        if self.verbose:
                                            logging.info(f"  + SUCCESS! Retrieved {len(extract)} characters from {lang_name} Wikipedia")
                                            logging.info(f"  + Page: {sanitize_for_console(page_title)}")
                                            logging.info(f"  + URL: {page_url}")
                                        return extract, page_title, page_url

                            if self.verbose and search_term == search_variations[-1]:
                                logging.info(f"       No content available for this page")

                except Exception as e:
                    if self.verbose:
                        logging.error(f"       Error searching {lang_name} Wikipedia: {e}")
                    continue  # Try next variation

        # PASS 2: If original didn't work, try variations across all languages
        if other_variations:
            if self.verbose:
                logging.info(f"  > Original name not found. Trying {len(other_variations)} variations across all languages...")

            for lang_code, lang_name in languages:
                for search_term in other_variations:
                    try:
                        if self.verbose:
                            logging.info(f"    -> Trying variation '{sanitize_for_console(search_term)}' in {lang_name} Wikipedia...")

                        time.sleep(self.rate_limit_delay)

                        # Use Wikipedia API to search
                        search_url = f"https://{lang_code}.wikipedia.org/w/api.php"
                        search_params = {
                            'action': 'opensearch',
                            'search': search_term,
                            'limit': 5,  # Get more results to choose from
                            'format': 'json'
                        }

                        response = self.session.get(search_url, params=search_params, timeout=10)

                        if response.status_code == 200:
                            results = response.json()
                            if len(results) > 1 and len(results[1]) > 0:
                                # results[1] contains titles, results[3] contains URLs
                                titles = results[1]
                                urls = results[3] if len(results) > 3 else []

                                # Try to find the best match (same logic as Pass 1)
                                best_match_idx = None
                                brand_keywords = ['brand', 'company', 'product', 'chocolate', 'candy',
                                                'food', 'beverage', 'corporation', 'manufacturer',
                                                'soft drink', 'soda', 'bar', 'snack', 'retailer',
                                                'chain', 'store']

                                # Keywords that indicate NON-brand pages (to exclude)
                                exclude_keywords = ['film', 'movie', 'television', 'tv series', 'novel',
                                                  'book', 'album', 'song', 'band', 'actor', 'actress',
                                                  'director', 'singer', 'musician', 'character',
                                                  'video game', 'game', 'play', 'opera', 'anime',
                                                  'manga', 'comic', 'series', 'episode', 'season']

                                for idx, title in enumerate(titles):
                                    title_lower = title.lower()

                                    # Skip if this looks like a non-brand page
                                    if any(exclude_word in title_lower for exclude_word in exclude_keywords):
                                        if self.verbose:
                                            logging.info(f"       Skipping non-brand result: '{sanitize_for_console(title)}'")
                                        continue

                                    brand_name_clean = re.sub(r'[^\w\s]', '', brand_name).lower()
                                    title_clean = re.sub(r'[^\w\s]', '', title).lower()

                                    if brand_name_clean in title_clean:
                                        if any(keyword in title_lower for keyword in brand_keywords):
                                            best_match_idx = idx
                                            break
                                        elif best_match_idx is None:
                                            best_match_idx = idx

                                # If no specific match found, try to find first non-excluded result
                                if best_match_idx is None:
                                    for idx, title in enumerate(titles):
                                        title_lower = title.lower()
                                        if not any(exclude_word in title_lower for exclude_word in exclude_keywords):
                                            best_match_idx = idx
                                            break

                                # If still no match (all results were excluded), skip this variation
                                if best_match_idx is None:
                                    if self.verbose:
                                        logging.info(f"       All results filtered out as non-brand pages")
                                    continue  # Try next variation

                                page_title = titles[best_match_idx]
                                page_url = urls[best_match_idx] if best_match_idx < len(urls) else None

                                if self.verbose:
                                    logging.info(f"       Found match: '{sanitize_for_console(page_title)}'")
                                    logging.info(f"       (matched using variation: '{sanitize_for_console(search_term)}')")
                                    logging.info(f"       Fetching page content...")

                                # Get page content
                                content_params = {
                                    'action': 'query',
                                    'titles': page_title,
                                    'prop': 'extracts|revisions',
                                    'explaintext': True,
                                    'rvprop': 'content',
                                    'format': 'json'
                                }

                                content_response = self.session.get(search_url, params=content_params, timeout=10)

                                if content_response.status_code == 200:
                                    data = content_response.json()
                                    pages = data.get('query', {}).get('pages', {})

                                    for page_id, page_data in pages.items():
                                        extract = page_data.get('extract', '')
                                        if extract:
                                            if self.verbose:
                                                logging.info(f"  + SUCCESS! Retrieved {len(extract)} characters from {lang_name} Wikipedia")
                                                logging.info(f"  + Page: {sanitize_for_console(page_title)}")
                                                logging.info(f"  + URL: {page_url}")
                                            return extract, page_title, page_url

                    except Exception as e:
                        if self.verbose:
                            logging.error(f"       Error searching {lang_name} Wikipedia: {e}")
                        continue  # Try next variation

        if self.verbose:
            total_attempts = len(languages) + (len(languages) * len(other_variations) if other_variations else 0)
            logging.info(f"  - FAILED: No Wikipedia page found ({total_attempts} total attempts across {len(languages)} languages)")
        return None, None, None

    def search_duckduckgo(self, query: str, max_results: int = 5) -> str:
        """
        Fallback: Search DuckDuckGo and get combined text from snippets

        Note: DuckDuckGo may block automated searches. This is a best-effort fallback.
        The primary data source is Wikipedia.
        """
        try:
            if self.verbose:
                logging.info(f"  > Searching DuckDuckGo (fallback): '{sanitize_for_console(query)}'")
            time.sleep(self.rate_limit_delay)

            # Create fresh DDGS instance to reduce blocking
            try:
                from ddgs import DDGS  # New package name (v9.x)
            except ImportError:
                from duckduckgo_search import DDGS  # Fallback to old package name

            # Use context manager and try multiple approaches
            results = []

            # Try 1: With safesearch='off' (sometimes works better)
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, backend='bing', max_results=max_results, safesearch='off'))
            except Exception:
                pass

            # Try 2: Without safesearch parameter if first attempt failed
            if not results:
                try:
                    time.sleep(1)  # Small delay between attempts
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, backend='bing', max_results=max_results))
                except Exception:
                    pass

            # Combine all result snippets
            snippets = []
            for result in results:
                body = result.get('body', '')
                if body:
                    snippets.append(body)

            combined = ' '.join(snippets)
            if self.verbose:
                if results:
                    logging.info(f"  + Got DuckDuckGo results ({len(combined)} chars from {len(results)} results)")
                else:
                    logging.warning(f"  - DuckDuckGo returned 0 results (likely blocked - this is a known issue)")
                    logging.warning(f"  - Consider this brand as having no additional data available")
            # Sanitize the combined text to remove Unicode issues
            return sanitize_for_console(combined)
        except Exception as e:
            if self.verbose:
                logging.error(f"  X DuckDuckGo search error: {e}")
            return ""

    def extract_country(self, text: str) -> Optional[str]:
        """Extract country name from text"""
        if not text:
            return None

        # Common country names to search for
        countries = [
            'United States', 'USA', 'U.S.', 'America',
            'United Kingdom', 'UK', 'Britain', 'England',
            'Germany', 'France', 'Italy', 'Spain', 'Portugal',
            'Japan', 'China', 'South Korea', 'India', 'Taiwan',
            'Canada', 'Mexico', 'Brazil', 'Argentina',
            'Australia', 'New Zealand',
            'Switzerland', 'Sweden', 'Norway', 'Denmark', 'Finland',
            'Netherlands', 'Belgium', 'Austria', 'Ireland',
            'Russia', 'Poland', 'Czech Republic',
            'Singapore', 'Malaysia', 'Thailand', 'Indonesia'
        ]

        # Normalize country names
        country_map = {
            'USA': 'United States',
            'U.S.': 'United States',
            'America': 'United States',
            'UK': 'United Kingdom',
            'Britain': 'United Kingdom',
            'England': 'United Kingdom'
        }

        text_lower = text.lower()

        for country in countries:
            if country.lower() in text_lower:
                return country_map.get(country, country)

        return None

    def extract_year(self, text: str) -> Optional[int]:
        """Extract founding year from text"""
        if not text:
            return None

        # Year patterns
        patterns = [
            r'founded\s+(?:in\s+)?(\d{4})',
            r'established\s+(?:in\s+)?(\d{4})',
            r'introduced\s+(?:in\s+)?(\d{4})',
            r'launched\s+(?:in\s+)?(\d{4})',
            r'created\s+(?:in\s+)?(\d{4})',
            r'started\s+(?:in\s+)?(\d{4})',
            r'\((\d{4})\)',  # Year in parentheses near beginning
        ]

        # Check first 1000 characters for founding info
        text_snippet = text[:1000]

        for pattern in patterns:
            match = re.search(pattern, text_snippet, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 1700 <= year <= 2025:
                    return year

        return None

    def extract_employees(self, text: str) -> Optional[int]:
        """Extract employee count from text"""
        if not text:
            return None

        # Employee patterns
        patterns = [
            r'(\d{1,3}(?:,\d{3})+)\s+employees',
            r'(\d+,?\d*)\s+employees',
            r'employees[:\s]+(\d{1,3}(?:,\d{3})+)',
            r'employs\s+(?:over|about|approximately)?\s*(\d{1,3}(?:,\d{3})+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                emp_str = match.group(1).replace(',', '')
                try:
                    return int(emp_str)
                except ValueError:
                    continue

        return None

    def extract_market_cap(self, text: str) -> Optional[float]:
        """Extract market capitalization in billion USD"""
        if not text:
            return None

        patterns = [
            r'market\s+cap(?:italization)?[:\s]+\$?(\d+(?:\.\d+)?)\s*billion',
            r'valued\s+at\s+\$?(\d+(?:\.\d+)?)\s*billion',
            r'\$(\d+(?:\.\d+)?)\s*billion\s+market\s+cap'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def extract_esg_sustainability(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Extract ESG programs and sustainability actions"""
        if not text:
            return None, None

        text_lower = text.lower()

        # ESG program keywords
        esg_keywords = [
            'net zero', 'carbon neutral', 'renewable energy',
            'emissions reduction', 'sustainable agriculture',
            'circular economy', 'biodiversity', 'ESG',
            'environmental, social and governance',
            'sustainability report', 'climate commitment'
        ]

        # Sustainability action keywords
        sustainability_keywords = [
            'recycling', 'renewable', 'solar', 'wind power',
            'packaging reduction', 'water conservation',
            'regenerative agriculture', 'sustainable sourcing',
            'zero waste', 'carbon offset'
        ]

        found_esg = [kw for kw in esg_keywords if kw in text_lower]
        found_sustainability = [kw for kw in sustainability_keywords if kw in text_lower]

        esg_programs = '; '.join(found_esg[:3]) if found_esg else None
        sustainability_actions = '; '.join(found_sustainability[:3]) if found_sustainability else None

        return esg_programs, sustainability_actions

    def extract_fossil_fuel_reliance(self, text: str) -> Optional[str]:
        """Determine fossil fuel reliance level"""
        if not text:
            return None

        text_lower = text.lower()

        # Check for indicators
        if any(kw in text_lower for kw in ['100% renewable', 'fully renewable', 'no fossil fuels']):
            return 'Low'
        elif any(kw in text_lower for kw in ['renewable energy', 'solar', 'wind', 'hydroelectric']):
            return 'Medium'
        elif any(kw in text_lower for kw in ['coal', 'natural gas', 'petroleum', 'oil company']):
            return 'High'

        return None

    def extract_target_population(self, text: str) -> Optional[str]:
        """Extract target population/demographics"""
        if not text:
            return None

        text_lower = text.lower()

        # Demographic indicators
        targets = []

        if any(kw in text_lower for kw in ['premium', 'luxury', 'high-end']):
            targets.append('affluent consumers')
        if any(kw in text_lower for kw in ['budget', 'affordable', 'value']):
            targets.append('value-conscious consumers')
        if any(kw in text_lower for kw in ['young', 'millennial', 'gen z']):
            targets.append('young adults')
        if any(kw in text_lower for kw in ['family', 'families', 'children']):
            targets.append('families')
        if any(kw in text_lower for kw in ['professional', 'business']):
            targets.append('professionals')

        return ', '.join(targets) if targets else None

    def extract_rd_spend(self, text: str) -> Optional[float]:
        """Extract R&D spending as percentage of revenue"""
        if not text:
            return None

        patterns = [
            r'R&D.*?(\d+(?:\.\d+)?)%',
            r'research and development.*?(\d+(?:\.\d+)?)%\s+of\s+revenue',
            r'(\d+(?:\.\d+)?)%.*?R&D'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def extract_women_board_percent(self, text: str) -> Optional[float]:
        """Extract percentage of women on board"""
        if not text:
            return None

        patterns = [
            r'(\d+(?:\.\d+)?)%\s+(?:of\s+)?(?:board|directors).*?women',
            r'women.*?(\d+(?:\.\d+)?)%\s+(?:of\s+)?board',
            r'board.*?(\d+(?:\.\d+)?)%\s+women'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def extract_ceo_tenure(self, text: str) -> Optional[int]:
        """Extract CEO tenure in years"""
        if not text:
            return None

        patterns = [
            r'CEO\s+since\s+(\d{4})',
            r'chief executive officer\s+since\s+(\d{4})',
            r'appointed\s+CEO\s+in\s+(\d{4})'
        ]

        current_year = datetime.now().year

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    start_year = int(match.group(1))
                    if 1950 <= start_year <= current_year:
                        return current_year - start_year
                except ValueError:
                    continue

        return None

    def extract_customer_loyalty_index(self, text: str) -> Optional[float]:
        """
        Extract or estimate customer loyalty index (0-100)
        This is a rough estimate based on mentions of loyalty, satisfaction, etc.
        """
        if not text:
            return None

        text_lower = text.lower()

        # Score based on positive indicators
        score = 50  # baseline

        if 'high customer satisfaction' in text_lower or 'customer loyalty' in text_lower:
            score += 20
        if 'award' in text_lower and 'customer' in text_lower:
            score += 10
        if 'repeat customers' in text_lower or 'loyal customer' in text_lower:
            score += 10
        if 'poor customer' in text_lower or 'customer complaints' in text_lower:
            score -= 20

        # Look for specific loyalty metrics
        patterns = [
            r'loyalty.*?(\d+)%',
            r'retention.*?(\d+)%',
            r'satisfaction.*?(\d+)%'
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # Return estimated score if no specific metric found
        return min(max(score, 0), 100)

    def update_brand_fields(self, brand_name: str) -> Optional[Dict[str, any]]:
        """
        Update all specified fields for a brand using Wikipedia (primary) and DuckDuckGo (fallback)
        Returns dictionary with field updates, or None if no data found
        """
        logging.info("="*60)
        logging.info(f"Processing: {sanitize_for_console(brand_name)}")
        logging.info("="*60)

        # Get Wikipedia page
        wiki_content, page_title, page_url = self.get_wikipedia_page(brand_name)

        # If no Wikipedia page, try DuckDuckGo as fallback
        if not wiki_content:
            logging.info(f"  - No Wikipedia page found, trying DuckDuckGo fallback...")
            ddg_content = self.search_duckduckgo(f"When was {brand_name} founded? headquarters? employees?")
            if not ddg_content:
                logging.info(f"  - No data found via Wikipedia or DuckDuckGo for {brand_name} - skipping")
                return None
            # Use DuckDuckGo content instead
            wiki_content = ddg_content
            print(f"****\nContent:\n{wiki_content}\n****")
            page_url = None  # No specific Wikipedia URL from DDG

        # Initialize results dictionary (only store fields that we find data for)
        results = {}

        # Extract each field
        if self.verbose:
            logging.info("  > Extracting fields...")

        # Country of origin
        country_origin = self.extract_country(wiki_content[:500])
        if country_origin:
            results['country_of_origin'] = country_origin
            if self.verbose:
                logging.info(f"    - Country of origin: {country_origin}")

        # Year of foundation
        year = self.extract_year(wiki_content)
        if year:
            results['year_of_foundation'] = year
            if self.verbose:
                logging.info(f"    - Year of foundation: {year}")

        # Headquarters country (same as origin usually)
        hq_country = self.extract_country(wiki_content[:1000])
        if hq_country:
            results['headquarters_country'] = hq_country
            if self.verbose:
                logging.info(f"    - Headquarters country: {hq_country}")

        # Employees
        employees = self.extract_employees(wiki_content)
        if employees:
            results['employees'] = employees
            if self.verbose:
                logging.info(f"    - Employees: {employees:,}")

        # Fossil fuel reliance
        fossil_fuel = self.extract_fossil_fuel_reliance(wiki_content)
        if fossil_fuel:
            results['fossil_fuel_reliance'] = fossil_fuel
            if self.verbose:
                logging.info(f"    - Fossil fuel reliance: {fossil_fuel}")

        # ESG programs and sustainability actions
        esg, sustainability = self.extract_esg_sustainability(wiki_content)
        if esg:
            results['esg_programs'] = esg
            if self.verbose:
                logging.info(f"    - ESG programs: {sanitize_for_console(esg)}")
        if sustainability:
            results['sustainability_actions'] = sustainability
            if self.verbose:
                logging.info(f"    - Sustainability actions: {sanitize_for_console(sustainability)}")

        # Target population
        target_pop = self.extract_target_population(wiki_content)
        if target_pop:
            results['target_population'] = target_pop
            if self.verbose:
                logging.info(f"    - Target population: {target_pop}")

        # Customer loyalty index
        loyalty = self.extract_customer_loyalty_index(wiki_content)
        if loyalty and loyalty != 50:  # Only if not default
            results['customer_loyalty_index'] = loyalty
            if self.verbose:
                logging.info(f"    - Customer loyalty index: {loyalty}")

        # R&D spend percent
        rd_spend = self.extract_rd_spend(wiki_content)
        if rd_spend:
            results['r_and_d_spend_percent_revenue'] = rd_spend
            if self.verbose:
                logging.info(f"    - R&D spend % of revenue: {rd_spend}")

        # Women board percent
        women_board = self.extract_women_board_percent(wiki_content)
        if women_board:
            results['women_board_percent'] = women_board
            if self.verbose:
                logging.info(f"    - Women board %: {women_board}")

        # CEO tenure
        ceo_tenure = self.extract_ceo_tenure(wiki_content)
        if ceo_tenure:
            results['ceo_tenure_years'] = ceo_tenure
            if self.verbose:
                logging.info(f"    - CEO tenure (years): {ceo_tenure}")

        # Market cap
        market_cap = self.extract_market_cap(wiki_content)
        if market_cap:
            results['market_cap_billion_usd'] = market_cap
            if self.verbose:
                logging.info(f"    - Market cap (billion USD): {market_cap}")

        # References and links (only if we have a Wikipedia URL)
        if page_url:
            results['references_and_links'] = page_url
            if self.verbose:
                logging.info(f"    - Reference: {page_url}")
        elif results:  # If we found data but via DDG, note the source
            results['references_and_links'] = f"DuckDuckGo search: {brand_name}"
            if self.verbose:
                logging.info(f"    - Reference: DuckDuckGo search")

        # If no fields were found, return None to skip updating this record
        if not results:
            logging.info(f"  - No extractable data found for {brand_name} - skipping")
            return None

        if self.verbose:
            logging.info(f"  + Found {len(results)} fields with data")

        logging.info(f"[COMPLETED] {sanitize_for_console(brand_name)}")
        logging.info("")

        return results

    def process_csv(self, batch_size: int = 50, dry_run: bool = False, dry_run_count: int = 20):
        """Process CSV file and update fields"""
        # Load input CSV with encoding fallback
        try:
            # Try UTF-8 first
            df = pd.read_csv(self.input_csv, encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to Latin-1 (handles most special characters)
            try:
                logging.warning("UTF-8 decoding failed, trying Latin-1 encoding...")
                df = pd.read_csv(self.input_csv, encoding='latin-1')
            except Exception as e2:
                # Last resort: try Windows-1252
                try:
                    logging.warning("Latin-1 decoding failed, trying Windows-1252 encoding...")
                    df = pd.read_csv(self.input_csv, encoding='windows-1252')
                except Exception as e3:
                    logging.error(f"Failed to load input CSV with multiple encodings: {e3}")
                    sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to load input CSV: {e}")
            sys.exit(1)

        if self.brand_column not in df.columns:
            logging.error(f"Brand column '{self.brand_column}' not found in CSV")
            logging.error(f"Available columns: {', '.join(df.columns)}")
            sys.exit(1)

        total_rows = len(df)

        if dry_run:
            logging.info("="*80)
            logging.info("DRY RUN MODE - Processing random sample")
            logging.info("="*80)
            import random
            current_seed = int(time.time())
            random.seed(current_seed)
            sample_size = min(dry_run_count, total_rows)
            df = df.sample(n=sample_size, random_state=current_seed)
            logging.info(f"Selected {sample_size} random rows from {total_rows} total")
        else:
            logging.info("="*80)
            logging.info(f"FULL RUN MODE - Processing all {total_rows} rows")
            logging.info("="*80)

        # Ensure fields exist in dataframe (but don't initialize with N/A)
        for field in self.FIELDS_TO_UPDATE:
            if field not in df.columns:
                df[field] = None

        start_time = time.time()
        processed = 0
        skipped_no_wiki = 0

        # Process each row
        for idx, row in df.iterrows():
            brand_name = row[self.brand_column]

            if pd.isna(brand_name) or str(brand_name).strip() == '':
                logging.warning(f"  - Skipping row {idx}: empty brand name")
                continue

            try:
                # Get updated fields
                updated_fields = self.update_brand_fields(brand_name)

                # Only update if Wikipedia page was found
                if updated_fields is not None:
                    # Update dataframe
                    for field, value in updated_fields.items():
                        df.at[idx, field] = value
                    processed += 1
                else:
                    skipped_no_wiki += 1

                # Save checkpoint every batch_size rows
                if processed > 0 and processed % batch_size == 0:
                    checkpoint_file = f"{self.output_csv}.checkpoint_{processed}.csv"
                    df.to_csv(checkpoint_file, index=False, encoding='utf-8-sig')

                    elapsed = time.time() - start_time
                    avg_time = elapsed / processed
                    remaining = (len(df) - processed) * avg_time

                    logging.info("*"*80)
                    logging.info(f"CHECKPOINT SAVED: {checkpoint_file}")
                    logging.info(f"Progress: {processed}/{len(df)} updated ({100*processed/len(df):.1f}%)")
                    logging.info(f"Skipped (no Wikipedia): {skipped_no_wiki}")
                    logging.info(f"Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")
                    logging.info("*"*80)
                    logging.info("")

            except Exception as e:
                logging.error(f"  X Error processing {brand_name}: {e}")
                continue

        # Save final output with UTF-8 encoding (with BOM for Windows compatibility)
        df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')

        total_time = time.time() - start_time

        logging.info("="*80)
        logging.info("PROCESSING COMPLETE!")
        logging.info("="*80)
        logging.info(f"Total rows updated: {processed}/{len(df)}")
        logging.info(f"Skipped (no Wikipedia page): {skipped_no_wiki}")
        logging.info(f"Output file: {self.output_csv}")
        logging.info(f"Total time: {total_time/60:.1f} minutes")
        if processed > 0:
            logging.info(f"Average time per row: {total_time/processed:.1f} seconds")
        logging.info("="*80)

        return df

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Wikipedia Field Updater - Updates specific CSV fields using Wikipedia data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python wikipedia_field_updater.py --input_csv data.csv --output_csv data_updated.csv

  # Dry run with 10 random brands
  python wikipedia_field_updater.py --input_csv data.csv --output_csv test.csv --dry_run --dry_run_count 10

  # Custom brand column name
  python wikipedia_field_updater.py --input_csv data.csv --output_csv data_updated.csv --brand_column company_name

  # Slower rate limit (2 seconds between requests)
  python wikipedia_field_updater.py --input_csv data.csv --output_csv data_updated.csv --rate_limit 2.0
        """
    )

    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV file'
    )

    parser.add_argument(
        '--brand_column',
        type=str,
        default='brand_name',
        help='Name of the column containing brand names (default: brand_name)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Number of rows to process before saving checkpoint (default: 50)'
    )

    parser.add_argument(
        '--rate_limit',
        type=float,
        default=1.0,
        help='Seconds to wait between Wikipedia requests (default: 1.0)'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run in dry-run mode with a random sample'
    )

    parser.add_argument(
        '--dry_run_count',
        type=int,
        default=20,
        help='Number of random rows to process in dry-run mode (default: 20)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbosity (only show errors and major milestones)'
    )

    args = parser.parse_args()

    # Print configuration
    print("="*80)
    print("WIKIPEDIA FIELD UPDATER")
    print("="*80)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Brand column: {args.brand_column}")
    print(f"Batch size: {args.batch_size}")
    print(f"Rate limit: {args.rate_limit}s")
    print(f"Dry run: {args.dry_run}")
    if args.dry_run:
        print(f"Dry run sample size: {args.dry_run_count}")
    print(f"Verbose: {not args.quiet}")
    print("="*80)
    print()

    # Verify input file exists and is readable
    try:
        # Try UTF-8 first
        pd.read_csv(args.input_csv, nrows=1, encoding='utf-8')
    except UnicodeDecodeError:
        # Try with Latin-1 encoding
        try:
            pd.read_csv(args.input_csv, nrows=1, encoding='latin-1')
        except Exception:
            # Try with Windows-1252 encoding
            try:
                pd.read_csv(args.input_csv, nrows=1, encoding='windows-1252')
            except Exception as e:
                logging.error(f"Cannot read input file: {e}")
                sys.exit(1)
    except Exception as e:
        logging.error(f"Cannot read input file: {e}")
        sys.exit(1)

    # Initialize updater
    updater = WikipediaFieldUpdater(
        args.input_csv,
        args.output_csv,
        brand_column=args.brand_column,
        verbose=not args.quiet,
        rate_limit=args.rate_limit
    )

    # Process CSV
    updater.process_csv(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        dry_run_count=args.dry_run_count
    )

if __name__ == "__main__":
    main()
