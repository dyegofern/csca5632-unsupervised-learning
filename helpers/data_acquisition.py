"""
Data Acquisition Module
Handles loading company data and merging with brand relationship table
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import COMPANY_DATA_PATH, BRAND_MAPPING_PATH, VERBOSE


class DataAcquisition:
    """
    Loads and merges company ESG data with brand relationship table
    """

    def __init__(self, company_data_path: str = COMPANY_DATA_PATH,
                 brand_mapping_path: str = BRAND_MAPPING_PATH,
                 verbose: bool = VERBOSE):
        """
        Initialize data acquisition

        Args:
            company_data_path: Path to company ESG data
            brand_mapping_path: Path to brand-company mapping CSV
            verbose: Print progress messages
        """
        self.company_data_path = company_data_path
        self.brand_mapping_path = brand_mapping_path
        self.verbose = verbose

        self.company_df: Optional[pd.DataFrame] = None
        self.brand_mapping_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None

    def load_company_data(self, encoding_fallbacks: list = ['utf-8', 'latin-1', 'cp1252']) -> pd.DataFrame:
        """
        Load company ESG data with robust encoding handling

        Args:
            encoding_fallbacks: List of encodings to try

        Returns:
            DataFrame with company data
        """
        if self.verbose:
            print(f"Loading company data from: {self.company_data_path}")

        for encoding in encoding_fallbacks:
            try:
                self.company_df = pd.read_csv(self.company_data_path, encoding=encoding)
                if self.verbose:
                    print(f"✓ Successfully loaded with {encoding} encoding")
                    print(f"  Shape: {self.company_df.shape}")
                break
            except (UnicodeDecodeError, FileNotFoundError) as e:
                if encoding == encoding_fallbacks[-1]:
                    raise ValueError(f"Failed to load company data with all encodings: {e}")
                continue

        # Normalize column names
        self.company_df.columns = self.company_df.columns.str.lower().str.replace(' ', '_')

        return self.company_df

    def load_brand_mapping(self) -> pd.DataFrame:
        """
        Load brand-company relationship table
        Expected columns: brand_name, company_name (or similar identifiers)

        Returns:
            DataFrame with brand-company mapping
        """
        if self.verbose:
            print(f"Loading brand mapping from: {self.brand_mapping_path}")

        try:
            self.brand_mapping_df = pd.read_csv(self.brand_mapping_path)
            if self.verbose:
                print(f"✓ Successfully loaded brand mapping")
                print(f"  Shape: {self.brand_mapping_df.shape}")
                print(f"  Columns: {list(self.brand_mapping_df.columns)}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Brand mapping file not found: {self.brand_mapping_path}\n"
                "Please create this file with columns: brand_name, company_name"
            )

        # Normalize column names
        self.brand_mapping_df.columns = self.brand_mapping_df.columns.str.lower().str.replace(' ', '_')

        return self.brand_mapping_df

    def merge_data(self, company_key: str = 'company',
                   brand_company_key: str = 'company_name') -> pd.DataFrame:
        """
        Merge brand mapping with company data
        Each brand inherits parent company's ESG features

        Args:
            company_key: Column name in company_df to join on
            brand_company_key: Column name in brand_mapping_df to join on

        Returns:
            Merged DataFrame with brands and inherited company features
        """
        if self.company_df is None:
            self.load_company_data()
        if self.brand_mapping_df is None:
            self.load_brand_mapping()

        if self.verbose:
            print(f"\nMerging brand mapping with company data...")
            print(f"  Join keys: {brand_company_key} -> {company_key}")

        # Left join: Keep all brands, match with company data
        self.merged_df = self.brand_mapping_df.merge(
            self.company_df,
            left_on=brand_company_key,
            right_on=company_key,
            how='left',
            suffixes=('_brand', '_company')
        )

        if self.verbose:
            print(f"✓ Merge complete")
            print(f"  Merged shape: {self.merged_df.shape}")
            print(f"  Brands with company data: {self.merged_df[company_key].notna().sum()}")
            print(f"  Brands without company data: {self.merged_df[company_key].isna().sum()}")

        return self.merged_df

    def calculate_parent_esg_features(self, scope_col: str = 'scope1+2total',
                                     revenue_col: str = 'revenuesghgco') -> pd.DataFrame:
        """
        Calculate ESG score and risk for parent companies
        Uses same logic as supervised learning project

        Args:
            scope_col: Column name for emissions
            revenue_col: Column name for revenues

        Returns:
            DataFrame with parent ESG features added
        """
        if self.merged_df is None:
            raise ValueError("Must merge data first")

        if self.verbose:
            print("\nCalculating parent ESG features...")

        # Calculate emission-to-revenue ratio
        self.merged_df['parent_emission_ratio'] = (
            self.merged_df[scope_col] / self.merged_df[revenue_col]
        )

        # Apply log transformation
        self.merged_df['parent_esg_score_log'] = np.log1p(self.merged_df['parent_emission_ratio'])

        # Normalize to 0-100 scale
        min_log = self.merged_df['parent_esg_score_log'].min()
        max_log = self.merged_df['parent_esg_score_log'].max()
        self.merged_df['parent_esg_score'] = (
            ((self.merged_df['parent_esg_score_log'] - min_log) / (max_log - min_log)) * 100
        )

        # Classify parent risk (using percentiles)
        percentiles = self.merged_df['parent_esg_score'].quantile([0.2, 0.4, 0.6, 0.8])
        self.merged_df['parent_esg_risk'] = pd.cut(
            self.merged_df['parent_esg_score'],
            bins=[-np.inf, percentiles[0.2], percentiles[0.4],
                  percentiles[0.6], percentiles[0.8], np.inf],
            labels=['LOW', 'MEDIUM-LOW', 'MEDIUM', 'MEDIUM-HIGH', 'HIGH']
        )

        if self.verbose:
            print("✓ Parent ESG features calculated")
            print(f"  Parent Risk Distribution:")
            print(self.merged_df['parent_esg_risk'].value_counts().sort_index())

        return self.merged_df

    def get_summary(self) -> dict:
        """
        Get summary statistics of loaded and merged data

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'company_records': len(self.company_df) if self.company_df is not None else 0,
            'brand_records': len(self.brand_mapping_df) if self.brand_mapping_df is not None else 0,
            'merged_records': len(self.merged_df) if self.merged_df is not None else 0,
            'brands_per_company': None
        }

        if self.merged_df is not None:
            summary['brands_per_company'] = (
                self.merged_df.groupby('company_name')['brand_name']
                .count()
                .describe()
                .to_dict()
            )

        return summary


def create_sample_brand_mapping(output_path: str = BRAND_MAPPING_PATH,
                                company_names: list = None) -> pd.DataFrame:
    """
    Helper function to create a sample brand mapping CSV
    This is a template - user should replace with actual data

    Args:
        output_path: Where to save the CSV
        company_names: List of company names from the company dataset

    Returns:
        Sample DataFrame
    """
    sample_data = {
        'brand_name': [
            'Dove', 'Axe', 'Ben & Jerry\'s', 'Magnum', 'Knorr',
            'Hellmann\'s', 'Lipton', 'Breyers', 'Vaseline', 'Simple'
        ],
        'company_name': [
            'Unilever', 'Unilever', 'Unilever', 'Unilever', 'Unilever',
            'Unilever', 'Unilever', 'Unilever', 'Unilever', 'Unilever'
        ]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"Sample brand mapping created at: {output_path}")
    print("Please update this file with your actual brand-company relationships")

    return df


if __name__ == "__main__":
    # Example usage
    print("Data Acquisition Module - Example Usage\n")

    # Initialize
    data_acq = DataAcquisition()

    # Load company data
    company_df = data_acq.load_company_data()
    print(f"\nCompany data columns: {list(company_df.columns)[:10]}...")

    # Note: You need to create the brand mapping file first
    print("\n" + "="*60)
    print("NEXT STEP: Create brand_company_mapping.csv")
    print("="*60)
    print("Run: create_sample_brand_mapping() to generate template")
