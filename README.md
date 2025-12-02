# Unsupervised Learning Final Project
## Brand-Level ESG Risk Discovery: From Company Predictions to Brand Segmentation

**Course:** Introduction to Unsupervised Learning
**Institution:** University of Colorado Boulder
**Author:** Dyego Fernandes de Sousa

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation & Context](#motivation--context)
3. [The Problem Statement](#the-problem-statement)
4. [Why Unsupervised Learning?](#why-unsupervised-learning)
5. [Dataset](#dataset)
6. [Methodology](#methodology)
7. [Project Structure](#project-structure)
8. [Installation & Setup](#installation--setup)
9. [Usage Guide](#usage-guide)
10. [Key Findings](#key-findings)
11. [Technical Highlights](#technical-highlights)
12. [Future Work](#future-work)

---

## Project Overview

This project demonstrates the application of **unsupervised learning techniques** to discover hidden patterns in brand-level environmental, social, and governance (ESG) data. Building on a previous supervised learning project that predicted company-level ESG scores, this work explores the **non-linear relationship** between parent companies and their brands.

### Core Hypothesis

**Brands do not linearly inherit their parent company's ESG profile.**

A multinational corporation with high CO2 emissions might own brands marketed as "eco-friendly," and conversely, a company with strong ESG performance might own budget brands with poor environmental messaging.

---

## Motivation & Context

### Building on Supervised Learning

This project is a natural evolution from a supervised learning project where I:
- Predicted company-level ESG scores (regression)
- Classified sustainability risk categories (classification)
- Used company emissions and revenue data

### The Gap: What About Brands?

In reality, **consumers interact with brands, not parent companies.** A brand's environmental positioning can diverge significantly from its parent company's actual environmental impact, creating opportunities for:
- **Greenwashing**: High sustainability messaging with poor parent company performance
- **Hidden gems**: Truly sustainable brands under less-known parent companies
- **Market segmentation**: Understanding brand positioning patterns

---

## The Problem Statement

### Challenge

Given parent company ESG data and brand-specific attributes, **discover latent brand segments** that reveal the true relationship between brand messaging and parent company environmental impact.

### Goals

1. **Segment brands** into distinct groups using clustering algorithms
2. **Identify "greenwashing" candidates** - brands with high sustainability claims but polluting parent companies
3. **Discover anomalies** - brands that don't fit typical patterns
4. **Prove non-linearity** - demonstrate that brand positioning doesn't directly follow parent company risk
5. **Create actionable insights** for consumers, investors, and regulators

---

## Why Unsupervised Learning?

Unsupervised learning is the perfect approach for this problem because:

### 1. **No Ground Truth Labels**
We don't have pre-defined "greenwashing" labels or brand segment categories. Unsupervised learning discovers these patterns naturally from the data.

### 2. **Discovery-Oriented**
Unlike supervised learning (predict known outcomes), we're exploring **what patterns exist** in brand environmental positioning.

### 3. **Multi-dimensional Relationships**
Brands exist in a complex feature space combining:
- Parent company emissions and revenues
- Brand-specific sustainability messaging
- Market positioning (sector, price tier)
- Consumer perception (scraped data)

Clustering algorithms can identify groups in this high-dimensional space.

### 4. **Dimensionality Reduction**
With many features, visualization is impossible. PCA and t-SNE reduce dimensions while preserving relationships, enabling interpretable 2D/3D visualizations.

### 5. **Anomaly Detection**
DBSCAN identifies outlier brands that don't fit normal patterns - potentially the most interesting cases for investigation.

### 6. **Real-World Application**
This mirrors real scenarios where:
- Regulators need to identify potential greenwashing
- Investors want to understand brand portfolios
- Consumers seek truly sustainable brands
- Researchers study corporate environmental behavior

---

## Dataset

### Source Data

#### Company-Level Data
From the supervised learning project:
- **Source**: GHG Shopper and Stakeholder Takeover initiatives
- **Features**: Scope 1+2 emissions, revenues, ESG scores, risk categories
- **Size**: ~250 companies

#### Brand-Company Mapping
- **Format**: CSV with brand-to-company relationships
- **Source**: Manual compilation or public data
- **Purpose**: Link brands to parent companies

#### Brand-Specific Data (Web-Scraped)
- **Sector/Category**: Industry classification
- **Sustainability Keywords**: Count of environmental terms in brand descriptions
- **Mission Statements**: Text from brand websites/Wikipedia
- **Price Tier**: Budget, mid-range, premium, luxury
- **Source**: Wikipedia, brand websites

### Feature Engineering

#### Parent Company Features (Inherited)
- `parent_esg_score`: Company's environmental score (0-100)
- `parent_esg_risk`: Risk category (LOW, MEDIUM-LOW, MEDIUM, MEDIUM-HIGH, HIGH)
- `parent_emission_ratio`: Emissions per dollar of revenue
- `scope1+2total`: Total greenhouse gas emissions

#### Brand-Specific Features (Differentiating)
- `sustainability_keyword_count`: Frequency of environmental terms
- `sector`, `category`: Industry classification
- `price_tier_numeric`: Encoded price positioning

#### Engineered Features (Key Innovation)
- **`brand_green_score`**: Normalized sustainability messaging (0-100)
- **`brand_parent_divergence`**: Gap between brand image and parent reality
  - Formula: `brand_green_score - (100 - parent_pollution_score)`
  - Positive = brand appears greener than parent (potential greenwashing)
  - Negative = brand undersells parent's sustainability
- **`greenwashing_indicator`**: Binary flag for high divergence cases
- **`description_length`**, **`mission_length`**: Text complexity measures

---

## Methodology

### Pipeline Overview

```
1. Data Acquisition
   ├── Load company ESG data
   ├── Load brand-company mapping
   └── Merge and calculate parent features

2. Brand Differentiation (Web Scraping)
   ├── Scrape Wikipedia/websites
   ├── Extract sector, keywords, descriptions
   └── Merge with company data

3. Feature Engineering
   ├── Create divergence scores
   ├── Encode categorical variables (one-hot)
   ├── Handle missing values (median imputation)
   └── Scale features (StandardScaler)

4. Dimensionality Reduction
   ├── PCA: Reduce to 95% variance (~10-15 components)
   └── t-SNE: Reduce to 2D for visualization

5. Clustering
   ├── K-Means: Find optimal k with elbow method
   ├── Hierarchical: Create dendrogram, cut at k clusters
   └── DBSCAN: Density-based, identify outliers

6. Interpretation
   ├── Profile clusters (mean feature values)
   ├── Assign interpretable labels
   ├── Identify greenwashing candidates
   └── Analyze anomalies
```

### Algorithms Applied

#### K-Means Clustering
- **Why**: Fast, scalable, finds spherical clusters
- **Optimization**: Elbow method to determine k
- **Metrics**: Silhouette score, inertia, Calinski-Harabasz index

#### Hierarchical Clustering
- **Why**: Shows hierarchical relationships, no need to pre-specify k
- **Linkage**: Ward (minimizes within-cluster variance)
- **Visualization**: Dendrogram

#### DBSCAN
- **Why**: Finds arbitrary-shaped clusters, identifies outliers
- **Parameters**: Eps (distance threshold), min_samples
- **Output**: Core clusters + noise points (anomalies)

#### PCA (Principal Component Analysis)
- **Why**: Reduces dimensionality, removes noise, speeds up clustering
- **Variance**: Preserve 95% of information
- **Interpretation**: Analyze top features per component

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Why**: Creates interpretable 2D visualizations
- **Workflow**: Apply to PCA-reduced data (recommended for efficiency)
- **Use Case**: Final visualization for presentations

---

## Project Structure

```
unsupervised_learning/final_project/
│
├── README.md                          # This file
├── final.md                           # Original project requirements
│
├── unsupervised_final.ipynb           # Main Jupyter notebook (START HERE)
│
├── config.py                          # Configuration and constants
├── data_acquisition.py                # Load and merge data
├── web_scraper.py                     # Scrape brand data
├── feature_engineering.py             # Feature creation and scaling
├── clustering_models.py               # K-Means, Hierarchical, DBSCAN
├── dimensionality_reduction.py        # PCA, t-SNE
├── visualization_utils.py             # Plotting functions
│
├── data/                              # Data directory
│   ├── brand_company_mapping.csv      # Brand-to-company relationships (YOU CREATE THIS)
│   └── brand_data.csv                 # Scraped brand data (generated)
│
├── output/                            # Results directory
│   └── brand_clustering_results.csv   # Final clustered data
│
└── models/                            # Saved models (optional)
```

### Module Descriptions

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Central configuration | Paths, hyperparameters, keywords |
| `data_acquisition.py` | Data loading and merging | `load_company_data()`, `merge_data()`, `calculate_parent_esg_features()` |
| `web_scraper.py` | Web scraping utilities | `scrape_brand_data()`, `scrape_multiple_brands()` |
| `feature_engineering.py` | Feature creation | `create_divergence_score()`, `prepare_features_for_clustering()` |
| `clustering_models.py` | Clustering algorithms | `kmeans_clustering()`, `hierarchical_clustering()`, `dbscan_clustering()` |
| `dimensionality_reduction.py` | PCA and t-SNE | `apply_pca()`, `apply_tsne()`, `pca_then_tsne()` |
| `visualization_utils.py` | Plotting | `plot_clusters_2d()`, `plot_elbow_curve()`, `plot_dendrogram()` |

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install beautifulsoup4 requests scipy
```

### Setup Steps

1. **Clone or download this project**
   ```bash
   cd unsupervised_learning/final_project
   ```

2. **Create the brand-company mapping file**
   - Run the helper function in notebook or:
   ```python
   from data_acquisition import create_sample_brand_mapping
   create_sample_brand_mapping()
   ```
   - Edit `data/brand_company_mapping.csv` with your actual brand relationships

3. **Ensure company data is accessible**
   - Verify path in `config.py` points to your company ESG data
   - Default: `../../supervised_learning/ghg_data.csv`

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook unsupervised_final.ipynb
   ```

---

## Usage Guide

### Quick Start

1. **Open `unsupervised_final.ipynb`**
2. **Run cells sequentially** - the notebook is designed as a complete workflow
3. **Key decision points:**
   - Cell for web scraping: May take time, consider manual data entry alternative
   - Cluster labeling: Requires domain interpretation based on your data

### Customization

#### Adjust Clustering Parameters

In `config.py`:
```python
CLUSTERING_CONFIG = {
    'kmeans': {
        'n_clusters_range': range(2, 11),  # Try k from 2 to 10
        ...
    },
    'hierarchical': {
        'n_clusters': 5,  # Change default cluster count
        ...
    }
}
```

#### Change Feature Sets

In notebook (Phase 3):
```python
full_df, scaled_features, feature_names = engineer.prepare_features_for_clustering(
    df=full_df,
    parent_features=['parent_esg_score', 'scope1+2total'],  # Customize
    brand_features=['sustainability_keyword_count', 'price_tier_numeric'],  # Customize
    categorical_features=['sector', 'parent_esg_risk']  # Customize
)
```

#### Skip Web Scraping

Use manual data entry instead:
```python
# In notebook, Phase 2
ManualBrandEnrichment.create_template(
    brands=brand_names,
    output_path='data/brand_manual_data.csv'
)
# Fill in the CSV, then:
scraped_df = ManualBrandEnrichment.load_manual_data('data/brand_manual_data.csv')
```

---

## Key Findings

### Finding 1: Non-Linear Relationship Confirmed

**Result:** Brand clusters contain a mix of parent company risk levels.

**Evidence:** Cross-tabulation shows each brand cluster contains brands from multiple parent risk categories, proving brand positioning is independent of parent company ESG performance.

### Finding 2: Greenwashing Identification

**Method:** Brands with:
- High `sustainability_keyword_count` (>50th percentile)
- High parent `parent_esg_risk` (MEDIUM-HIGH or HIGH)

**Example Insights:**
- "Eco-friendly" brands owned by high-emission conglomerates
- Quantifiable divergence score enables ranking suspects

### Finding 3: Market Segmentation

**Discovered Clusters** (example - actual clusters depend on data):
1. **Sustainable Leaders**: Low parent risk + high sustainability messaging
2. **Greenwashing Suspects**: High parent risk + high sustainability messaging
3. **Pure Polluters**: High parent risk + low sustainability messaging
4. **Middle Ground**: Mixed characteristics
5. **Emerging Eco-Brands**: Small companies with strong environmental focus

### Finding 4: Anomaly Detection

**DBSCAN Outliers**: Brands that don't fit any cluster pattern
- Unique business models
- Extreme values on key features
- Potential data quality issues or genuinely anomalous cases

---

## Technical Highlights

### 1. Robust Data Pipeline
- Multiple encoding fallbacks for CSV loading
- Comprehensive missing value handling
- Graceful failure with informative error messages

### 2. Feature Engineering Innovation
- **Divergence score** quantifies brand-parent gap
- Combines inherited (parent) and differentiating (brand) features
- Scales diverse feature types appropriately

### 3. Multi-Algorithm Comparison
- K-Means: Optimal k via elbow method
- Hierarchical: Visual dendrogram analysis
- DBSCAN: Automatic outlier detection
- Quantitative comparison via silhouette, CH, DB scores

### 4. Dimensionality Reduction Best Practices
- **PCA first** (noise reduction, speed)
- **t-SNE second** (visualization)
- Feature importance analysis per principal component

### 5. Modular Design
- Clean separation of concerns
- Reusable components
- Easy to extend or modify algorithms

### 6. Comprehensive Visualization
- 2D/3D cluster plots
- Elbow curves
- Dendrograms
- Variance explained plots
- Feature importance
- Cluster profile heatmaps
- Algorithm comparison side-by-side

---

## Expected Deliverables

✅ **One Jupyter Notebook**: Complete workflow from data to insights
✅ **Modular Python Files**: Reusable components for each phase
✅ **Clustering Models**: K-Means, Hierarchical, DBSCAN implementations
✅ **Dimensionality Reduction**: PCA and t-SNE with visualizations
✅ **Cluster Profiles**: Interpretable segment descriptions
✅ **Visualizations**: Publication-ready plots
✅ **Comparative Analysis**: Algorithm performance comparison
✅ **Greenwashing Detection**: Quantifiable divergence scores
✅ **Documentation**: This comprehensive README

---

## Limitations & Considerations

### Data Quality
- Web scraping success depends on data availability
- Brand-company mapping requires manual curation
- Small sample sizes may limit statistical power

### Web Scraping
- Respects robots.txt and rate limits
- May fail for some brands (404s, paywalls)
- Consider manual data entry as fallback

### Clustering Interpretation
- Unsupervised learning requires domain knowledge to interpret clusters
- Cluster labels are subjective but data-driven
- Different random seeds can affect t-SNE visualizations

### Scalability
- Current implementation handles hundreds of brands efficiently
- For thousands of brands, consider mini-batch K-Means or sampling
- Dendrogram visualization becomes cluttered with >100 samples

---

## Future Work

### Data Enhancements
- **Social media sentiment analysis**: Twitter, Instagram brand perception
- **News article analysis**: Recent sustainability controversies or achievements
- **Product-level data**: Link specific products to brands
- **Temporal analysis**: Track brand positioning changes over time

### Methodological Extensions
- **Topic modeling**: LDA on mission statements to identify themes
- **Semi-supervised learning**: If partial labels become available
- **Network analysis**: Relationships between brands in same clusters
- **Time-series clustering**: Temporal patterns in brand messaging

### Applications
- **Recommendation system**: Suggest truly sustainable brands to consumers
- **Investment tool**: Identify ESG leaders and laggards for portfolio construction
- **Regulatory assistance**: Automated greenwashing detection for authorities
- **Academic research**: Study corporate environmental communication strategies

---

## References

### Datasets
- GHG Shopper: https://www.ghgshopper.org
- Stakeholder Takeover: https://www.stakeholdertakeover.org

### Techniques
- K-Means: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Hierarchical Clustering: Johnson, S. C. (1967). "Hierarchical clustering schemes"
- DBSCAN: Ester, M., et al. (1996). "A density-based algorithm for discovering clusters"
- PCA: Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space"
- t-SNE: van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t-SNE"

### Related Work
- Supervised Learning Project: `../../supervised_learning/supervised_learning_final.ipynb`

---

## Contact & Acknowledgments

**Author**: Dyego Fernandes de Sousa
**Course**: Introduction to Unsupervised Learning
**Institution**: University of Colorado Boulder

**Special Thanks**:
- Prof. Lynn M. LoPucki (UCLA Law School) for dataset permission
- Course instructors and TAs
- Colleagues who provided feedback on project direction

---

## License

This project is for educational purposes as part of university coursework.

---

## Appendix: Algorithm Complexity

| Algorithm | Time Complexity | Space Complexity | Best For |
|-----------|-----------------|------------------|----------|
| K-Means | O(n·k·i·d) | O(n·d) | Large datasets, spherical clusters |
| Hierarchical | O(n²·log n) | O(n²) | Small datasets, dendrogram visualization |
| DBSCAN | O(n·log n) | O(n) | Arbitrary shapes, outlier detection |
| PCA | O(min(n·d², d³)) | O(d²) | Dimensionality reduction, feature selection |
| t-SNE | O(n²) | O(n²) | Visualization only (slow for large n) |

*n = samples, d = dimensions, k = clusters, i = iterations*

---

**End of README**

For questions or issues, please refer to the notebook comments or reach out to the author.
