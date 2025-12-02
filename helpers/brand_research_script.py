"""
Comprehensive Brand Research Script using Wikipedia + DuckDuckGo
Processes 3,000+ brands with 45+ data columns per brand
Author: AI Assistant
Date: 2025-11-22
"""

import pandas as pd
import time
import json
import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from typing import Dict, List, Optional
import logging
from datetime import datetime
import argparse
import sys
import urllib.parse

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brand_research.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def sanitize_for_console(text: str) -> str:
    """Remove non-ASCII characters to prevent encoding errors on Windows console"""
    if not isinstance(text, str):
        text = str(text)
    # Replace common Unicode characters and then remove any remaining non-ASCII
    replacements = {
        '\u2192': '->',  # Right arrow
        '\u2713': '+',   # Check mark
        '\u2717': 'X',   # Cross mark
        '\u2298': '-',   # Circle with stroke
        '\ufffd': '?',   # Replacement character
        '\u2014': '--',  # Em dash
        '\u2013': '-',   # En dash
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    # Remove any remaining non-ASCII characters
    text = text.encode('ascii', errors='replace').decode('ascii')
    return text

class BrandResearcher:
    """Intelligent brand research using Wikipedia API + DuckDuckGo"""

    def __init__(self, input_csv: str, output_csv: str, verbose: bool = True):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.cache = {}
        self.rate_limit_delay = 1  # seconds between requests
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.ddgs = DDGS()

        logging.info("="*80)
        logging.info("Initializing Brand Researcher (Wikipedia API + DuckDuckGo)")
        logging.info("="*80)
        logging.info(f"Input CSV: {input_csv}")
        logging.info(f"Output CSV: {output_csv}")
        logging.info(f"Verbose mode: {verbose}")
        logging.info(f"Rate limit delay: {self.rate_limit_delay}s between requests")
        logging.info("Note: Using Wikipedia API as primary source, DuckDuckGo as fallback")

        # Define all required columns (matching dataset structure)
        self.columns = [
            # Industry columns (will be preserved from input)
            'industry_id', 'industry_name', 'deforestation_risk', 'labor_exploitation_risk',
            'chemical_pollution_risk', 'supply_chain_greenwashing_risk', 'positive_innovation_risk',
            'esg_summary',
            # Company columns (will be preserved from input)
            'company_id', 'company_name', 'scope_v_revenues', 'reported_scope',
            'emissions_rank', 'in_industry_rank', 'global_rank', 'exclusions',
            'initial_greenwashing_level', 'accusation_factor_1', 'accusation',
            'revenues', 'scope12_total',
            # Brand ID columns (will be preserved from input)
            'brand_id', 'brand_name', 'id',
            # Brand research columns (these will be populated by research)
            'brand_id', 'brand_name', 'country_of_origin', 'year_of_foundation',
            'headquarters_country', 'parent_company', 'employees',
            'age_prenatal', 'age_0_5', 'age_6_12', 'age_teens',
            'age_young_adults', 'age_seniors',
            'demographics_income_level', 'income_low', 'income_middle',
            'income_high', 'income_premium', 'income_snap_support',
            'demographics_geographic_reach', 'demographics_gender',
            'demographics_lifestyle', 'lifestyle_family', 'lifestyle_youth',
            'lifestyle_seniors', 'lifestyle_health_focused', 'lifestyle_convenience',
            'lifestyle_tech_savvy', 'lifestyle_sustainability_conscious',
            'online_sales', 'has_franchises', 'has_drive_through', 'owns_fleet',
            'electric_vehicles_percent', 'fossil_fuel_reliance', 'esg_programs',
            'sustainability_actions', 'target_population', 'main_partnerships',
            'branding_innovation_level', 'supply_chain_localization',
            'product_portfolio_diversity', 'revenue_billion_usd',
            'customer_loyalty_index', 'r_and_d_spend_percent_revenue',
            'women_board_percent', 'ceo_tenure_years', 'market_cap_billion_usd',
            'major_sustainability_award_last5y', 'references_and_links'
        ]

    def get_wikipedia_page(self, brand_name: str) -> Optional[str]:
        """Get Wikipedia page content for a brand using Wikipedia API"""
        try:
            if self.verbose:
                logging.info(f"  > Searching Wikipedia for: '{sanitize_for_console(brand_name)}'")
            time.sleep(self.rate_limit_delay)

            # Use Wikipedia API to search
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                'action': 'opensearch',
                'search': brand_name,
                'limit': 3,
                'format': 'json'
            }

            response = self.session.get(search_url, params=search_params, timeout=10)

            if response.status_code == 200:
                results = response.json()
                if len(results) > 1 and len(results[1]) > 0:
                    page_title = results[1][0]  # First search result

                    # Get page content
                    content_params = {
                        'action': 'query',
                        'titles': page_title,
                        'prop': 'extracts',
                        'explaintext': True,
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
                                    logging.info(f"  + Found Wikipedia page: {sanitize_for_console(page_title)}")
                                return extract

            if self.verbose:
                logging.info(f"  - No Wikipedia page found")
            return None

        except Exception as e:
            if self.verbose:
                logging.error(f"  X Wikipedia error for '{brand_name}': {e}")
            return None

    def search_duckduckgo(self, query: str, max_results: int = 5) -> str:
        """Fallback: Search DuckDuckGo and get snippets"""
        try:
            if self.verbose:
                logging.info(f"  > Searching DuckDuckGo (fallback): '{sanitize_for_console(query)}'")
            time.sleep(self.rate_limit_delay)

            results = list(self.ddgs.text(query, max_results=max_results))

            # Combine all result snippets
            snippets = []
            for result in results:
                body = result.get('body', '')
                if body:
                    snippets.append(body)

            combined = ' '.join(snippets)
            if self.verbose:
                logging.info(f"  + Got DuckDuckGo results ({len(combined)} chars from {len(results)} results)")
            # Sanitize the combined text to remove Unicode issues
            return sanitize_for_console(combined)
        except Exception as e:
            if self.verbose:
                logging.error(f"  X DuckDuckGo search error: {e}")
            return ""

    def extract_parent_company(self, brand_name: str, wiki_content: Optional[str] = None) -> Optional[str]:
        """Find parent company for brand"""
        if self.verbose:
            logging.info(f"  [1/6] Extracting parent company information...")

        # Try Wikipedia first
        text = wiki_content if wiki_content else ""

        # If no Wikipedia, try DuckDuckGo
        if not text:
            text = self.search_duckduckgo(f"{brand_name} brand parent company owner")

        # Parse for parent company mentions
        parent_keywords = ['owned by', 'subsidiary of', 'part of', 'division of', 'acquired by', 'brand of']
        text_lower = text.lower()

        for keyword in parent_keywords:
            if keyword in text_lower:
                # Extract company name after keyword
                match = re.search(f"{keyword}\s+([A-Z][\w\s&]+(?:Inc|Corp|Ltd|LLC|Co|Company)?)", text)
                if match:
                    parent = match.group(1).strip()
                    if self.verbose:
                        logging.info(f"  + Parent company found: {sanitize_for_console(parent)}")
                    return parent

        if self.verbose:
            logging.info(f"  - No parent company identified")
        return None

    def extract_founding_info(self, brand_name: str, wiki_content: Optional[str] = None) -> Dict[str, any]:
        """Extract founding year, country, and headquarters"""
        if self.verbose:
            logging.info(f"  [2/6] Extracting founding information...")

        info = {
            'year_of_foundation': None,
            'country_of_origin': None,
            'headquarters_country': None
        }

        # Try Wikipedia first
        text = wiki_content if wiki_content else ""

        # If no Wikipedia, try DuckDuckGo
        if not text:
            text = self.search_duckduckgo(f"{brand_name} brand founded year headquarters country")

        # Extract year - look for "founded" or "established" followed by year
        year_patterns = [
            r'founded\s+(?:in\s+)?(\d{4})',
            r'established\s+(?:in\s+)?(\d{4})',
            r'introduced\s+(?:in\s+)?(\d{4})',
            r'launched\s+(?:in\s+)?(\d{4})',
            r'\((\d{4})\)',  # Year in parentheses
        ]

        for pattern in year_patterns:
            year_match = re.search(pattern, text, re.IGNORECASE)
            if year_match:
                year = int(year_match.group(1))
                if 1800 <= year <= 2025:
                    info['year_of_foundation'] = year
                    break

        # Extract country names
        countries = ['United States', 'USA', 'UK', 'United Kingdom', 'Switzerland', 'Germany',
                    'France', 'Japan', 'China', 'India', 'Brazil', 'Canada',
                    'Mexico', 'Italy', 'Spain', 'Netherlands', 'Belgium', 'Australia',
                    'South Korea', 'Sweden', 'Denmark', 'Norway', 'Finland']

        for country in countries:
            if country.lower() in text.lower():
                if not info['country_of_origin']:
                    info['country_of_origin'] = country
                if not info['headquarters_country']:
                    info['headquarters_country'] = country
                break

        if self.verbose:
            year = info['year_of_foundation']
            origin = sanitize_for_console(str(info['country_of_origin'])) if info['country_of_origin'] else 'None'
            hq = sanitize_for_console(str(info['headquarters_country'])) if info['headquarters_country'] else 'None'
            logging.info(f"  + Year: {year}, Origin: {origin}, HQ: {hq}")

        return info

    def extract_company_size(self, brand_name: str, parent_company: str) -> Dict[str, any]:
        """Extract employee count and revenue"""
        if self.verbose:
            logging.info(f"  [3/6] Extracting company size information...")
        search_name = parent_company if parent_company else brand_name
        if self.verbose and parent_company:
            logging.info(f"  > Searching using parent company: {sanitize_for_console(parent_company)}")

        query = f"{search_name} employees revenue annual report"
        text = self.search_duckduckgo(query)

        info = {
            'employees': None,
            'revenue_billion_usd': None
        }

        # Extract employee count (look for numbers with 'employees' nearby)
        emp_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*(?:thousand|million)?\s*employees',
                            text, re.IGNORECASE)
        if emp_match:
            emp_str = emp_match.group(1).replace(',', '')
            if 'million' in text[emp_match.start():emp_match.end()].lower():
                info['employees'] = int(float(emp_str) * 1000000)
            elif 'thousand' in text[emp_match.start():emp_match.end()].lower():
                info['employees'] = int(float(emp_str) * 1000)
            else:
                info['employees'] = int(emp_str)

        # Extract revenue (in billions)
        rev_match = re.search(r'\$?(\d+(?:\.\d+)?)\s*(?:billion|bn)', text, re.IGNORECASE)
        if rev_match:
            info['revenue_billion_usd'] = float(rev_match.group(1))

        if self.verbose:
            logging.info(f"  + Employees: {info['employees']}, Revenue: ${info['revenue_billion_usd']}B")

        return info

    def extract_demographics(self, brand_name: str, parent_company: str) -> Dict[str, any]:
        """Extract target demographics and market positioning"""
        if self.verbose:
            logging.info(f"  [4/6] Extracting demographics and target market...")
        query = f"{brand_name} target market demographics consumer profile"
        combined_text = self.search_duckduckgo(query).lower()

        demo = {
            'age_prenatal': 0, 'age_0_5': 0, 'age_6_12': 0,
            'age_teens': 0, 'age_young_adults': 0, 'age_seniors': 0,
            'income_low': 0, 'income_middle': 0, 'income_high': 0,
            'income_premium': 0, 'income_snap_support': 0,
            'lifestyle_family': 0, 'lifestyle_youth': 0, 'lifestyle_seniors': 0,
            'lifestyle_health_focused': 0, 'lifestyle_convenience': 0,
            'lifestyle_tech_savvy': 0, 'lifestyle_sustainability_conscious': 0
        }

        # Age group detection
        age_keywords = {
            'age_prenatal': ['prenatal', 'pregnancy', 'expecting'],
            'age_0_5': ['infant', 'baby', 'toddler', 'preschool'],
            'age_6_12': ['children', 'kids', 'elementary'],
            'age_teens': ['teen', 'adolescent', 'youth', '13-17', '12-18'],
            'age_young_adults': ['young adult', 'millennial', 'gen z', '18-35', '20-40'],
            'age_seniors': ['senior', 'elderly', 'retiree', '65+', 'older adult']
        }

        for age_group, keywords in age_keywords.items():
            if any(kw in combined_text for kw in keywords):
                demo[age_group] = 1

        # Income level detection
        if any(kw in combined_text for kw in ['affordable', 'budget', 'low-cost', 'economic']):
            demo['income_low'] = 1
        if any(kw in combined_text for kw in ['middle class', 'mainstream', 'average']):
            demo['income_middle'] = 1
        if any(kw in combined_text for kw in ['upscale', 'high-end', 'wealthy']):
            demo['income_high'] = 1
        if any(kw in combined_text for kw in ['premium', 'luxury', 'exclusive']):
            demo['income_premium'] = 1

        # Lifestyle detection
        if any(kw in combined_text for kw in ['family', 'families', 'parents']):
            demo['lifestyle_family'] = 1
        if any(kw in combined_text for kw in ['health', 'wellness', 'fitness', 'organic']):
            demo['lifestyle_health_focused'] = 1
        if any(kw in combined_text for kw in ['convenience', 'quick', 'easy', 'on-the-go']):
            demo['lifestyle_convenience'] = 1
        if any(kw in combined_text for kw in ['tech', 'digital', 'app', 'online']):
            demo['lifestyle_tech_savvy'] = 1
        if any(kw in combined_text for kw in ['sustainable', 'eco', 'green', 'environment']):
            demo['lifestyle_sustainability_conscious'] = 1

        if self.verbose:
            age_targets = [k.replace('age_', '') for k, v in demo.items() if k.startswith('age_') and v == 1]
            income_targets = [k.replace('income_', '') for k, v in demo.items() if k.startswith('income_') and v == 1]
            lifestyle_targets = [k.replace('lifestyle_', '') for k, v in demo.items() if k.startswith('lifestyle_') and v == 1]
            logging.info(f"  + Age groups: {', '.join(age_targets) if age_targets else 'None identified'}")
            logging.info(f"  + Income levels: {', '.join(income_targets) if income_targets else 'None identified'}")
            logging.info(f"  + Lifestyles: {', '.join(lifestyle_targets) if lifestyle_targets else 'None identified'}")

        return demo

    def extract_operations(self, brand_name: str, parent_company: str) -> Dict[str, any]:
        """Extract operational details"""
        if self.verbose:
            logging.info(f"  [5/6] Extracting operational details...")
        query = f"{brand_name} franchise operations fleet delivery distribution"
        combined_text = self.search_duckduckgo(query).lower()

        ops = {
            'has_franchises': 0,
            'has_drive_through': 0,
            'owns_fleet': 0,
            'online_sales': 0
        }

        if any(kw in combined_text for kw in ['franchise', 'franchisee', 'franchising']):
            ops['has_franchises'] = 1
        if any(kw in combined_text for kw in ['drive-through', 'drive thru', 'drive-thru']):
            ops['has_drive_through'] = 1
        if any(kw in combined_text for kw in ['fleet', 'delivery vehicles', 'trucks', 'distribution']):
            ops['owns_fleet'] = 1
        if any(kw in combined_text for kw in ['online', 'e-commerce', 'website', 'buy online']):
            ops['online_sales'] = 1

        if self.verbose:
            features = []
            if ops['has_franchises']: features.append('franchises')
            if ops['has_drive_through']: features.append('drive-through')
            if ops['owns_fleet']: features.append('delivery fleet')
            if ops['online_sales']: features.append('online sales')
            logging.info(f"  + Operations: {', '.join(features) if features else 'No special features identified'}")

        return ops

    def extract_sustainability(self, brand_name: str, parent_company: str) -> Dict[str, any]:
        """Extract sustainability and ESG information"""
        if self.verbose:
            logging.info(f"  [6/6] Extracting sustainability and ESG data...")
        search_name = parent_company if parent_company else brand_name
        if self.verbose and parent_company:
            logging.info(f"  > Searching using parent company: {sanitize_for_console(parent_company)}")
        query = f"{search_name} sustainability ESG renewable energy electric vehicles"
        combined_text = self.search_duckduckgo(query)

        sustain = {
            'electric_vehicles_percent': 0,
            'fossil_fuel_reliance': 'Unknown',
            'esg_programs': '',
            'sustainability_actions': '',
            'major_sustainability_award_last5y': 0
        }

        # Extract EV percentage
        ev_match = re.search(r'(\d+)%\s*(?:electric|EV)', combined_text, re.IGNORECASE)
        if ev_match:
            sustain['electric_vehicles_percent'] = int(ev_match.group(1))

        # ESG programs
        esg_keywords = ['net zero', 'carbon neutral', 'renewable energy', 'emissions reduction',
                       'sustainable agriculture', 'circular economy', 'biodiversity']
        found_programs = [kw for kw in esg_keywords if kw in combined_text.lower()]
        sustain['esg_programs'] = '; '.join(found_programs[:3]) if found_programs else 'Limited public info'

        # Sustainability actions
        action_keywords = ['recycling', 'renewable', 'packaging', 'water conservation',
                          'regenerative', 'solar', 'wind power']
        found_actions = [kw for kw in action_keywords if kw in combined_text.lower()]
        sustain['sustainability_actions'] = '; '.join(found_actions[:3]) if found_actions else 'Limited public info'

        # Awards
        if any(kw in combined_text.lower() for kw in ['award', 'recognition', 'certified', 'b corp']):
            sustain['major_sustainability_award_last5y'] = 1

        if self.verbose:
            logging.info(f"  + EV%: {sustain['electric_vehicles_percent']}, Awards: {sustain['major_sustainability_award_last5y']}")
            esg_preview = sanitize_for_console(sustain['esg_programs'][:50])
            logging.info(f"  + ESG programs: {esg_preview}...")

        return sustain

    def research_brand(self, brand_id: str, brand_name: str, existing_data: Dict = None) -> Dict[str, any]:
        """Complete research pipeline for a single brand"""
        logging.info("="*60)
        logging.info(f"RESEARCHING: {sanitize_for_console(brand_name)} (ID: {brand_id})")
        logging.info("="*60)

        # Start with existing data (industry, company columns) if provided
        brand_data = existing_data.copy() if existing_data else {}

        # Ensure brand identifiers are set
        brand_data.update({
            'brand_id': brand_id,
            'brand_name': brand_name
        })

        # Step 0: Get Wikipedia page content (use for all extractions)
        wiki_content = self.get_wikipedia_page(brand_name)

        # Step 1: Find parent company
        parent_company = self.extract_parent_company(brand_name, wiki_content)
        brand_data['parent_company'] = parent_company

        # Step 2: Founding information
        founding_info = self.extract_founding_info(brand_name, wiki_content)
        brand_data.update(founding_info)

        # Step 3: Company size
        size_info = self.extract_company_size(brand_name, parent_company)
        brand_data.update(size_info)

        # Step 4: Demographics
        demo_info = self.extract_demographics(brand_name, parent_company)
        brand_data.update(demo_info)

        # Step 5: Operations
        ops_info = self.extract_operations(brand_name, parent_company)
        brand_data.update(ops_info)

        # Step 6: Sustainability
        sustain_info = self.extract_sustainability(brand_name, parent_company)
        brand_data.update(sustain_info)

        # Fill remaining columns with defaults
        defaults = {
            'demographics_income_level': 'Mixed',
            'demographics_geographic_reach': 'Regional to Global',
            'demographics_gender': 'All genders',
            'demographics_lifestyle': 'Varied',
            'target_population': 'General consumers',
            'main_partnerships': 'Research in progress',
            'branding_innovation_level': 1,
            'supply_chain_localization': 30,
            'product_portfolio_diversity': 1,
            'customer_loyalty_index': 50,
            'r_and_d_spend_percent_revenue': 1.0,
            'women_board_percent': 25,
            'ceo_tenure_years': 3,
            'market_cap_billion_usd': 0,
            'references_and_links': f"https://duckduckgo.com/?q={brand_name.replace(' ', '+')}"
        }

        for key, value in defaults.items():
            if key not in brand_data:
                brand_data[key] = value

        logging.info(f"[COMPLETED] {sanitize_for_console(brand_name)}")
        logging.info("")

        return brand_data

    def process_all_brands(self, batch_size: int = 50, dry_run: bool = False, dry_run_count: int = 20):
        """Process all brands in batches with checkpointing"""
        # Load input
        df_input = pd.read_csv(self.input_csv)
        total_brands = len(df_input)

        if dry_run:
            logging.info("="*80)
            logging.info("DRY RUN MODE - Processing random sample of brands")
            logging.info("="*80)
            # Randomly sample brands using current time as seed for different results each run
            import random
            current_seed = int(time.time())
            random.seed(current_seed)
            sample_size = min(dry_run_count, total_brands)
            df_input = df_input.sample(n=sample_size, random_state=current_seed)
            logging.info(f"Selected {sample_size} random brands from {total_brands} total brands")
            logging.info(f"Random seed: {current_seed} (different brands each run)")
        else:
            logging.info("="*80)
            logging.info(f"FULL RUN MODE - Processing all {total_brands} brands")
            logging.info("="*80)

        logging.info(f"Loaded {len(df_input)} brands from {self.input_csv}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Output file: {self.output_csv}")
        logging.info("")

        # Initialize output dataframe
        results = []

        start_time = time.time()

        # Process in batches
        for i in range(0, len(df_input), batch_size):
            batch = df_input.iloc[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(df_input) + batch_size - 1) // batch_size

            logging.info("*"*80)
            logging.info(f"BATCH {batch_num}/{total_batches}: Processing brands {i+1} to {min(i+batch_size, len(df_input))}")
            logging.info("*"*80)

            for _, row in batch.iterrows():
                try:
                    # Convert row to dictionary to preserve existing columns
                    existing_data = row.to_dict()
                    brand_data = self.research_brand(
                        row['brand_id'],
                        row['brand_name'],
                        existing_data=existing_data
                    )
                    results.append(brand_data)
                except Exception as e:
                    logging.error("="*60)
                    logging.error(f"ERROR processing {sanitize_for_console(row['brand_name'])}: {sanitize_for_console(str(e))}")
                    logging.error("="*60)
                    # Add row data with error marker
                    error_data = row.to_dict()
                    error_data['error'] = str(e)
                    results.append(error_data)

            # Checkpoint after each batch
            df_results = pd.DataFrame(results)
            checkpoint_file = f"{self.output_csv}.checkpoint_{i}.csv"
            df_results.to_csv(checkpoint_file, index=False)

            elapsed_time = time.time() - start_time
            avg_time_per_brand = elapsed_time / len(results)
            remaining_brands = len(df_input) - len(results)
            estimated_remaining = avg_time_per_brand * remaining_brands

            logging.info("*"*80)
            logging.info(f"CHECKPOINT SAVED: {checkpoint_file}")
            logging.info(f"Progress: {len(results)}/{len(df_input)} brands ({100*len(results)/len(df_input):.1f}%)")
            logging.info(f"Elapsed time: {elapsed_time/60:.1f} minutes")
            logging.info(f"Avg time per brand: {avg_time_per_brand:.1f} seconds")
            logging.info(f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
            logging.info("*"*80)
            logging.info("")

        # Final save
        df_final = pd.DataFrame(results)
        df_final.to_csv(self.output_csv, index=False)

        total_time = time.time() - start_time

        logging.info("="*80)
        logging.info("PROCESSING COMPLETE!")
        logging.info("="*80)
        logging.info(f"Total brands processed: {len(df_final)}")
        logging.info(f"Output file: {self.output_csv}")
        logging.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        logging.info(f"Average time per brand: {total_time/len(df_final):.1f} seconds")
        logging.info("="*80)

        return df_final

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Brand Research Script using Wikipedia API + DuckDuckGo Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default input file
  python brand_research_script.py

  # Run with custom input file
  python brand_research_script.py --input_csv my_brands.csv

  # Dry run with 20 random brands (different each time)
  python brand_research_script.py --dry_run

  # Dry run with custom count (e.g., 50 brands)
  python brand_research_script.py --dry_run --dry_run_count 50

  # Run with custom batch size and quiet mode
  python brand_research_script.py --batch_size 100 --quiet
        """
    )

    parser.add_argument(
        '--input_csv',
        type=str,
        default='data/raw/brand_information.csv',
        help='Path to input CSV file with brands (default: data/raw/brand_information.csv)'
    )

    parser.add_argument(
        '--output_csv',
        type=str,
        default='data/raw/brand_information_updated.csv',
        help='Path to output CSV file (default: data/raw/brand_information_updated.csv)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Number of brands to process per batch (default: 50)'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run in dry-run mode with a random sample of brands'
    )

    parser.add_argument(
        '--dry_run_count',
        type=int,
        default=20,
        help='Number of random brands to process in dry-run mode (default: 20)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbosity (only show errors and major milestones)'
    )

    args = parser.parse_args()

    # Print configuration
    print("="*80)
    print("BRAND RESEARCH SCRIPT")
    print("="*80)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dry run: {args.dry_run}")
    if args.dry_run:
        print(f"Dry run sample size: {args.dry_run_count}")
    print(f"Verbose: {not args.quiet}")
    print("="*80)
    print()

    # Verify input file exists
    if not pd.io.common.file_exists(args.input_csv):
        logging.error(f"Input file not found: {args.input_csv}")
        sys.exit(1)

    # Initialize researcher
    researcher = BrandResearcher(
        args.input_csv,
        args.output_csv,
        verbose=not args.quiet
    )

    # Process brands
    researcher.process_all_brands(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        dry_run_count=args.dry_run_count
    )

if __name__ == "__main__":
    main()
