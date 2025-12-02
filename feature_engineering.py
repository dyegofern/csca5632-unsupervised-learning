"""
Feature Engineering Module
Creates brand-level features and calculates divergence/hypocrisy scores
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import FEATURE_CONFIG, VERBOSE


class BrandFeatureEngineer:
    """
    Engineers features for brand-level unsupervised learning
    """

    def __init__(self, config: dict = FEATURE_CONFIG, verbose: bool = VERBOSE):
        """
        Initialize feature engineer

        Args:
            config: Feature engineering configuration
            verbose: Print progress messages
        """
        self.config = config
        self.verbose = verbose

        # Scalers
        self.scaler = None
        self.imputer = None
        self.encoder = None

        # Feature sets
        self.numerical_features = []
        self.categorical_features = []
        self.engineered_features = []

    def create_divergence_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create "Hypocrisy" or "Divergence" score
        Measures gap between brand image (keywords) and parent company reality (ESG risk)

        Args:
            df: DataFrame with brand and parent features

        Returns:
            DataFrame with divergence score added
        """
        if self.verbose:
            print("Creating divergence/hypocrisy scores...")

        # Normalize sustainability keyword count (0-100 scale)
        if df['sustainability_keyword_count'].max() > 0:
            df['brand_green_score'] = (
                (df['sustainability_keyword_count'] / df['sustainability_keyword_count'].max()) * 100
            )
        else:
            df['brand_green_score'] = 0

        # Divergence = High green marketing + High parent risk
        # Negative score means brand appears greener than parent
        if 'parent_esg_score' in df.columns:
            # Invert parent ESG score (higher score = worse performance in original)
            df['parent_pollution_score'] = 100 - df['parent_esg_score']

            # Divergence: positive when brand claims green but parent is polluter
            df['brand_parent_divergence'] = df['brand_green_score'] - (100 - df['parent_pollution_score'])

            # Greenwashing indicator (high keywords, high parent risk)
            df['greenwashing_indicator'] = (
                (df['brand_green_score'] > 50) &
                (df['parent_pollution_score'] > 50)
            ).astype(int)

        if self.verbose:
            print("✓ Divergence scores created")
            if 'greenwashing_indicator' in df.columns:
                print(f"  Potential greenwashing brands: {df['greenwashing_indicator'].sum()}")

        self.engineered_features.extend([
            'brand_green_score',
            'brand_parent_divergence',
            'greenwashing_indicator'
        ])

        return df

    def encode_categorical_features(self, df: pd.DataFrame,
                                    categorical_cols: List[str]) -> pd.DataFrame:
        """
        One-hot encode categorical variables

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names

        Returns:
            DataFrame with encoded features
        """
        if self.verbose:
            print(f"\nEncoding {len(categorical_cols)} categorical features...")

        # Filter to existing columns
        existing_cats = [col for col in categorical_cols if col in df.columns]

        if not existing_cats:
            if self.verbose:
                print("  No categorical columns to encode")
            return df

        # Handle missing values in categorical columns
        for col in existing_cats:
            df[col] = df[col].fillna('Unknown')

        # One-hot encode
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = self.encoder.fit_transform(df[existing_cats])

        # Create column names
        encoded_cols = []
        for i, col in enumerate(existing_cats):
            categories = self.encoder.categories_[i]
            encoded_cols.extend([f"{col}_{cat}" for cat in categories])

        # Add to DataFrame
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoded_cols,
            index=df.index
        )

        df = pd.concat([df, encoded_df], axis=1)

        if self.verbose:
            print(f"✓ Encoding complete: {len(encoded_cols)} new features created")

        self.categorical_features = existing_cats

        return df

    def handle_missing_values(self, df: pd.DataFrame,
                             numerical_cols: List[str]) -> pd.DataFrame:
        """
        Impute missing values in numerical features

        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names

        Returns:
            DataFrame with imputed values
        """
        if self.verbose:
            print("\nHandling missing values...")

        existing_nums = [col for col in numerical_cols if col in df.columns]

        if not existing_nums:
            return df

        # Check for missing values
        missing_counts = df[existing_nums].isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if len(cols_with_missing) == 0:
            if self.verbose:
                print("  No missing values found")
            return df

        if self.verbose:
            print(f"  Columns with missing values: {len(cols_with_missing)}")
            print(f"  Imputation strategy: {self.config['handle_missing']}")

        # Apply imputation
        if self.config['handle_missing'] == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif self.config['handle_missing'] == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        else:
            # Drop rows with missing values
            df = df.dropna(subset=existing_nums)
            if self.verbose:
                print(f"  Dropped rows with missing values. New shape: {df.shape}")
            return df

        # Impute
        df[existing_nums] = self.imputer.fit_transform(df[existing_nums])

        if self.verbose:
            print(f"✓ Missing values imputed")

        return df

    def scale_features(self, df: pd.DataFrame,
                      feature_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Scale numerical features using StandardScaler or MinMaxScaler

        Args:
            df: Input DataFrame
            feature_cols: List of feature columns to scale

        Returns:
            Tuple of (original DataFrame, scaled feature array)
        """
        if self.verbose:
            print(f"\nScaling features using {self.config['scale_method']} scaler...")

        existing_features = [col for col in feature_cols if col in df.columns]

        if not existing_features:
            raise ValueError("No valid feature columns found for scaling")

        # Initialize scaler
        if self.config['scale_method'] == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        # Scale
        scaled_array = self.scaler.fit_transform(df[existing_features])

        if self.verbose:
            print(f"✓ Scaling complete")
            print(f"  Scaled features shape: {scaled_array.shape}")
            print(f"  Feature names: {existing_features[:5]}..." if len(existing_features) > 5 else f"  Feature names: {existing_features}")

        self.numerical_features = existing_features

        return df, scaled_array

    def engineer_price_tier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create numerical features from price tier information

        Args:
            df: DataFrame with price_tier column

        Returns:
            DataFrame with price tier features
        """
        if 'price_tier' not in df.columns:
            return df

        if self.verbose:
            print("Engineering price tier features...")

        # Map price tiers to numerical values
        price_map = {
            'budget': 1,
            'mid-range': 2,
            'mid range': 2,
            'premium': 3,
            'luxury': 4
        }

        df['price_tier_numeric'] = (
            df['price_tier']
            .str.lower()
            .map(price_map)
            .fillna(2)  # Default to mid-range
        )

        if self.verbose:
            print("✓ Price tier features created")

        self.engineered_features.append('price_tier_numeric')

        return df

    def create_brand_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features representing brand complexity
        (e.g., length of description, presence of certain attributes)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with complexity features
        """
        if self.verbose:
            print("Creating brand complexity features...")

        # Description length
        if 'description' in df.columns:
            df['description_length'] = df['description'].fillna('').str.len()
            self.engineered_features.append('description_length')

        # Mission statement length
        if 'mission_statement' in df.columns:
            df['mission_length'] = df['mission_statement'].fillna('').str.len()
            self.engineered_features.append('mission_length')

        # Has Wikipedia page
        if 'wikipedia_url' in df.columns:
            df['has_wikipedia'] = df['wikipedia_url'].notna().astype(int)
            self.engineered_features.append('has_wikipedia')

        if self.verbose:
            print(f"✓ Created {len([f for f in self.engineered_features if f in df.columns])} complexity features")

        return df

    def prepare_features_for_clustering(self, df: pd.DataFrame,
                                       parent_features: List[str] = None,
                                       brand_features: List[str] = None,
                                       categorical_features: List[str] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Complete feature engineering pipeline for clustering

        Args:
            df: Input DataFrame
            parent_features: List of parent company features to include
            brand_features: List of brand-specific features to include
            categorical_features: List of categorical features to encode

        Returns:
            Tuple of (DataFrame, scaled_feature_array, feature_names)
        """
        if self.verbose:
            print("="*60)
            print("FEATURE ENGINEERING PIPELINE")
            print("="*60)

        # Default feature sets
        if parent_features is None:
            parent_features = ['parent_esg_score', 'parent_emission_ratio', 'scope1+2total']

        if brand_features is None:
            brand_features = ['sustainability_keyword_count']

        if categorical_features is None:
            categorical_features = ['sector', 'category', 'price_tier', 'parent_esg_risk']

        # 1. Create divergence scores
        df = self.create_divergence_score(df)

        # 2. Engineer price tier
        df = self.engineer_price_tier_features(df)

        # 3. Create complexity features
        df = self.create_brand_complexity_features(df)

        # 4. Combine all numerical features
        all_numerical = (
            [f for f in parent_features if f in df.columns] +
            [f for f in brand_features if f in df.columns] +
            [f for f in self.engineered_features if f in df.columns]
        )

        # 5. Handle missing values
        df = self.handle_missing_values(df, all_numerical)

        # 6. Encode categorical features
        df = self.encode_categorical_features(df, categorical_features)

        # 7. Get encoded categorical column names
        encoded_cats = [col for col in df.columns if any(
            col.startswith(f"{cat}_") for cat in categorical_features
        )]

        # 8. Final feature list
        final_features = all_numerical + encoded_cats

        # 9. Scale features
        df, scaled_array = self.scale_features(df, final_features)

        if self.verbose:
            print("\n" + "="*60)
            print("FEATURE ENGINEERING COMPLETE")
            print("="*60)
            print(f"Total features: {len(final_features)}")
            print(f"  Numerical: {len(all_numerical)}")
            print(f"  Categorical (encoded): {len(encoded_cats)}")
            print(f"  Engineered: {len([f for f in self.engineered_features if f in df.columns])}")

        return df, scaled_array, final_features


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module - Example Usage\n")

    # Create sample data
    sample_data = pd.DataFrame({
        'brand_name': ['Brand A', 'Brand B', 'Brand C'],
        'parent_esg_score': [30, 70, 50],
        'parent_emission_ratio': [100, 500, 250],
        'sustainability_keyword_count': [15, 2, 8],
        'sector': ['Food', 'Automotive', 'Food'],
        'price_tier': ['premium', 'luxury', 'budget']
    })

    # Initialize engineer
    engineer = BrandFeatureEngineer()

    # Create divergence scores
    sample_data = engineer.create_divergence_score(sample_data)

    print("Sample data with divergence scores:")
    print(sample_data[['brand_name', 'brand_green_score', 'brand_parent_divergence', 'greenwashing_indicator']])
