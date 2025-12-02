"""
Configuration file for Unsupervised Learning Final Project
Brand-Level ESG Risk Analysis
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data file paths
COMPANY_DATA_PATH = os.path.join(PROJECT_ROOT, '../../supervised_learning/ghg_data.csv')
BRAND_MAPPING_PATH = os.path.join(DATA_DIR, 'brand_company_mapping.csv')  # To be created
BRAND_DATA_PATH = os.path.join(DATA_DIR, 'brand_data.csv')  # Generated after scraping

# Web scraping configuration
SCRAPING_CONFIG = {
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'delay_between_requests': 2,  # seconds
    'max_retries': 3,
    'timeout': 10
}

# Keywords for brand mission statement analysis
SUSTAINABILITY_KEYWORDS = [
    'sustainable', 'sustainability', 'green', 'eco-friendly', 'eco friendly',
    'organic', 'recycled', 'renewable', 'carbon neutral', 'net zero',
    'environment', 'environmental', 'climate', 'biodegradable',
    'ethical', 'responsible', 'conservation', 'planet'
]

# Feature engineering configuration
FEATURE_CONFIG = {
    'scale_method': 'standard',  # 'standard' or 'minmax'
    'handle_missing': 'median',  # 'median', 'mean', or 'drop'
    'min_samples_per_brand': 1
}

# Clustering configuration
CLUSTERING_CONFIG = {
    'kmeans': {
        'n_clusters_range': range(2, 11),
        'random_state': 42,
        'n_init': 10,
        'max_iter': 300
    },
    'hierarchical': {
        'n_clusters': 5,
        'linkage': 'ward',
        'distance_metric': 'euclidean'
    },
    'dbscan': {
        'eps_range': [0.3, 0.5, 0.7, 1.0],
        'min_samples': 5
    }
}

# Dimensionality reduction configuration
DIM_REDUCTION_CONFIG = {
    'pca': {
        'n_components': 0.95,  # Preserve 95% variance
        'random_state': 42
    },
    'tsne': {
        'n_components': 2,
        'perplexity': 30,
        'learning_rate': 200,
        'n_iter': 1000,
        'random_state': 42
    }
}

# Visualization configuration
VIZ_CONFIG = {
    'figsize': (12, 8),
    'dpi': 100,
    'style': 'whitegrid',
    'color_palette': 'viridis'
}

# Random state for reproducibility
RANDOM_STATE = 42

# Verbose logging
VERBOSE = True
