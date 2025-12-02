"""
Dimensionality Reduction Module
Implements PCA and t-SNE for visualization and feature reduction
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from config import DIM_REDUCTION_CONFIG, RANDOM_STATE, VERBOSE


class DimensionalityReducer:
    """
    Reduces feature dimensionality for visualization and noise reduction
    """

    def __init__(self, config: dict = DIM_REDUCTION_CONFIG,
                 random_state: int = RANDOM_STATE,
                 verbose: bool = VERBOSE):
        """
        Initialize dimensionality reducer

        Args:
            config: Dimensionality reduction configuration
            random_state: Random seed for reproducibility
            verbose: Print progress messages
        """
        self.config = config
        self.random_state = random_state
        self.verbose = verbose

        # Store models
        self.pca_model: Optional[PCA] = None
        self.tsne_model: Optional[TSNE] = None

        # Store transformed data
        self.pca_transformed: Optional[np.ndarray] = None
        self.tsne_transformed: Optional[np.ndarray] = None

        # Store explained variance
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.cumulative_variance_ratio: Optional[np.ndarray] = None

    def apply_pca(self, X: np.ndarray,
                  n_components: float = None) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction

        Args:
            X: Feature matrix (should be scaled)
            n_components: Number of components or variance to preserve (0-1 for variance)

        Returns:
            Tuple of (transformed data, fitted PCA model)
        """
        if self.verbose:
            print("="*60)
            print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
            print("="*60)

        if n_components is None:
            n_components = self.config['pca']['n_components']

        if self.verbose:
            if isinstance(n_components, float) and n_components < 1:
                print(f"Reducing dimensions while preserving {n_components*100}% variance...")
            else:
                print(f"Reducing to {n_components} components...")

        # Fit PCA
        self.pca_model = PCA(
            n_components=n_components,
            random_state=self.random_state
        )

        self.pca_transformed = self.pca_model.fit_transform(X)

        # Store variance information
        self.explained_variance_ratio = self.pca_model.explained_variance_ratio_
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)

        if self.verbose:
            print(f"✓ PCA complete")
            print(f"  Original dimensions: {X.shape[1]}")
            print(f"  Reduced dimensions: {self.pca_transformed.shape[1]}")
            print(f"  Total variance explained: {self.cumulative_variance_ratio[-1]:.4f}")
            print(f"\n  Variance per component:")
            for i, var in enumerate(self.explained_variance_ratio[:5], 1):
                print(f"    PC{i}: {var:.4f} ({var*100:.2f}%)")
            if len(self.explained_variance_ratio) > 5:
                print(f"    ... ({len(self.explained_variance_ratio) - 5} more components)")

        return self.pca_transformed, self.pca_model

    def apply_tsne(self, X: np.ndarray,
                   n_components: int = 2,
                   perplexity: int = None,
                   learning_rate: float = None) -> Tuple[np.ndarray, TSNE]:
        """
        Apply t-SNE for visualization

        Args:
            X: Feature matrix (should be scaled)
            n_components: Number of dimensions (typically 2 or 3 for visualization)
            perplexity: Perplexity parameter (balance local/global structure)
            learning_rate: Learning rate for optimization

        Returns:
            Tuple of (transformed data, fitted t-SNE model)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING (t-SNE)")
            print("="*60)

        # Use config defaults if not provided
        if perplexity is None:
            perplexity = self.config['tsne']['perplexity']
        if learning_rate is None:
            learning_rate = self.config['tsne']['learning_rate']

        if self.verbose:
            print(f"Reducing to {n_components}D for visualization...")
            print(f"Parameters:")
            print(f"  Perplexity: {perplexity}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Iterations: {self.config['tsne'].get('n_iter', 1000)}")

        # Fit t-SNE
        # Note: n_iter was renamed to max_iter in newer scikit-learn versions
        max_iter = self.config['tsne'].get('n_iter', self.config['tsne'].get('max_iter', 1000))

        self.tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )

        self.tsne_transformed = self.tsne_model.fit_transform(X)

        if self.verbose:
            print(f"✓ t-SNE complete")
            print(f"  Original dimensions: {X.shape[1]}")
            print(f"  Reduced dimensions: {self.tsne_transformed.shape[1]}")

        return self.tsne_transformed, self.tsne_model

    def pca_then_tsne(self, X: np.ndarray,
                      pca_components: float = 0.95,
                      tsne_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply PCA first (for noise reduction) then t-SNE (for visualization)
        This is a recommended workflow for high-dimensional data

        Args:
            X: Feature matrix
            pca_components: Variance to preserve in PCA
            tsne_components: Final dimensions for t-SNE

        Returns:
            Tuple of (PCA transformed data, t-SNE transformed data)
        """
        if self.verbose:
            print("="*60)
            print("TWO-STAGE DIMENSIONALITY REDUCTION: PCA → t-SNE")
            print("="*60)
            print("This approach is recommended for high-dimensional data:")
            print("1. PCA reduces noise and computational cost")
            print("2. t-SNE creates interpretable 2D/3D visualizations")
            print("="*60 + "\n")

        # Step 1: PCA
        pca_data, _ = self.apply_pca(X, n_components=pca_components)

        # Step 2: t-SNE on PCA-reduced data
        tsne_data, _ = self.apply_tsne(pca_data, n_components=tsne_components)

        return pca_data, tsne_data

    def get_top_features_per_component(self, feature_names: List[str],
                                      n_top: int = 5) -> pd.DataFrame:
        """
        Get top contributing features for each principal component

        Args:
            feature_names: Names of original features
            n_top: Number of top features to return per component

        Returns:
            DataFrame with top features per component
        """
        if self.pca_model is None:
            raise ValueError("PCA must be fitted first")

        if len(feature_names) != self.pca_model.components_.shape[1]:
            raise ValueError(f"Expected {self.pca_model.components_.shape[1]} feature names, got {len(feature_names)}")

        # Get components
        components = self.pca_model.components_

        # Create DataFrame for each component
        top_features_data = []

        for i, component in enumerate(components, 1):
            # Get absolute values (both positive and negative contributions matter)
            abs_component = np.abs(component)

            # Get top feature indices
            top_indices = np.argsort(abs_component)[-n_top:][::-1]

            # Create entries
            for rank, idx in enumerate(top_indices, 1):
                top_features_data.append({
                    'Component': f'PC{i}',
                    'Rank': rank,
                    'Feature': feature_names[idx],
                    'Loading': component[idx],
                    'Abs_Loading': abs_component[idx]
                })

        return pd.DataFrame(top_features_data)

    def get_variance_dataframe(self) -> pd.DataFrame:
        """
        Get variance explained by each principal component

        Returns:
            DataFrame with variance information
        """
        if self.pca_model is None:
            raise ValueError("PCA must be fitted first")

        return pd.DataFrame({
            'Component': [f'PC{i}' for i in range(1, len(self.explained_variance_ratio) + 1)],
            'Variance_Explained': self.explained_variance_ratio,
            'Cumulative_Variance': self.cumulative_variance_ratio
        })

    def determine_optimal_components(self, variance_threshold: float = 0.95) -> int:
        """
        Determine optimal number of components to preserve variance threshold

        Args:
            variance_threshold: Minimum cumulative variance to preserve

        Returns:
            Number of components needed
        """
        if self.cumulative_variance_ratio is None:
            raise ValueError("PCA must be fitted first")

        n_components = np.argmax(self.cumulative_variance_ratio >= variance_threshold) + 1

        if self.verbose:
            print(f"To preserve {variance_threshold*100}% variance:")
            print(f"  Need {n_components} components (out of {len(self.cumulative_variance_ratio)})")
            print(f"  Actual variance preserved: {self.cumulative_variance_ratio[n_components-1]:.4f}")

        return n_components

    def transform_new_data(self, X: np.ndarray, method: str = 'pca') -> np.ndarray:
        """
        Transform new data using fitted models

        Args:
            X: New feature matrix (must have same features as training data)
            method: 'pca' or 'tsne'

        Returns:
            Transformed data
        """
        if method == 'pca':
            if self.pca_model is None:
                raise ValueError("PCA model must be fitted first")
            return self.pca_model.transform(X)
        elif method == 'tsne':
            # Note: t-SNE cannot transform new data, must refit
            raise ValueError("t-SNE cannot transform new data. Must refit on combined dataset.")
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_reconstruction_error(self, X_original: np.ndarray,
                                X_pca: np.ndarray = None) -> float:
        """
        Calculate reconstruction error for PCA

        Args:
            X_original: Original data (scaled)
            X_pca: PCA-transformed data (if None, uses stored transformed data)

        Returns:
            Mean squared reconstruction error
        """
        if self.pca_model is None:
            raise ValueError("PCA must be fitted first")

        if X_pca is None:
            X_pca = self.pca_transformed

        # Reconstruct original data
        X_reconstructed = self.pca_model.inverse_transform(X_pca)

        # Calculate MSE
        mse = np.mean((X_original - X_reconstructed) ** 2)

        if self.verbose:
            print(f"PCA Reconstruction Error (MSE): {mse:.6f}")

        return mse


if __name__ == "__main__":
    # Example usage
    print("Dimensionality Reduction Module - Example Usage\n")

    # Generate sample high-dimensional data
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=100, n_features=20, centers=4, random_state=42)

    # Create feature names
    feature_names = [f'Feature_{i}' for i in range(1, 21)]

    # Initialize reducer
    reducer = DimensionalityReducer()

    # Apply PCA
    X_pca, _ = reducer.apply_pca(X, n_components=0.95)

    # Get variance information
    print("\n" + "="*60)
    print("VARIANCE EXPLAINED")
    print("="*60)
    print(reducer.get_variance_dataframe().head(10))

    # Get top features per component
    print("\n" + "="*60)
    print("TOP FEATURES PER COMPONENT")
    print("="*60)
    top_features = reducer.get_top_features_per_component(feature_names, n_top=3)
    print(top_features[top_features['Component'].isin(['PC1', 'PC2', 'PC3'])])

    # Apply t-SNE
    X_tsne, _ = reducer.apply_tsne(X_pca, n_components=2)

    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION COMPLETE")
    print("="*60)
    print(f"Original shape: {X.shape}")
    print(f"After PCA: {X_pca.shape}")
    print(f"After t-SNE: {X_tsne.shape}")
