"""
Clustering Models Module
Implements K-Means, Hierarchical Clustering, and DBSCAN for brand segmentation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Optional, Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
from itertools import product
from sklearn.model_selection import ParameterGrid

from config import CLUSTERING_CONFIG, RANDOM_STATE, VERBOSE


class BrandClusterer:
    """
    Performs unsupervised clustering to discover brand segments
    """

    def __init__(self, config: dict = CLUSTERING_CONFIG,
                 random_state: int = RANDOM_STATE,
                 verbose: bool = VERBOSE):
        """
        Initialize brand clusterer

        Args:
            config: Clustering configuration
            random_state: Random seed for reproducibility
            verbose: Print progress messages
        """
        self.config = config
        self.random_state = random_state
        self.verbose = verbose

        # Store models
        self.kmeans_model: Optional[KMeans] = None
        self.hierarchical_model: Optional[AgglomerativeClustering] = None
        self.dbscan_model: Optional[DBSCAN] = None

        # Store results
        self.cluster_labels: Dict[str, np.ndarray] = {}
        self.cluster_metrics: Dict[str, Dict] = {}
        self.elbow_data: Optional[pd.DataFrame] = None

    def kmeans_clustering(self, X: np.ndarray,
                         n_clusters: int = None,
                         find_optimal: bool = True) -> Tuple[KMeans, np.ndarray]:
        """
        Perform K-Means clustering

        Args:
            X: Feature matrix (scaled)
            n_clusters: Number of clusters (if None and find_optimal=True, uses elbow method)
            find_optimal: Whether to find optimal k using elbow method

        Returns:
            Tuple of (fitted model, cluster labels)
        """
        if self.verbose:
            print("="*60)
            print("K-MEANS CLUSTERING")
            print("="*60)

        # Find optimal k if requested
        if find_optimal and n_clusters is None:
            if self.verbose:
                print("Finding optimal number of clusters using elbow method...")
            n_clusters = self._find_optimal_k(X)

        if n_clusters is None:
            n_clusters = 5  # Default

        if self.verbose:
            print(f"Training K-Means with k={n_clusters}...")

        # Train K-Means
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=self.config['kmeans']['n_init'],
            max_iter=self.config['kmeans']['max_iter']
        )

        labels = self.kmeans_model.fit_predict(X)
        self.cluster_labels['kmeans'] = labels

        # Calculate metrics
        metrics = self._calculate_metrics(X, labels, 'K-Means')
        self.cluster_metrics['kmeans'] = metrics

        if self.verbose:
            print(f"✓ K-Means clustering complete")
            self._print_metrics(metrics)
            self._print_cluster_distribution(labels)

        return self.kmeans_model, labels

    def hierarchical_clustering(self, X: np.ndarray,
                               n_clusters: int = None) -> Tuple[AgglomerativeClustering, np.ndarray]:
        """
        Perform Hierarchical (Agglomerative) clustering

        Args:
            X: Feature matrix (scaled)
            n_clusters: Number of clusters

        Returns:
            Tuple of (fitted model, cluster labels)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("HIERARCHICAL CLUSTERING")
            print("="*60)

        if n_clusters is None:
            n_clusters = self.config['hierarchical']['n_clusters']

        if self.verbose:
            print(f"Training Hierarchical clustering with n_clusters={n_clusters}...")
            print(f"Linkage: {self.config['hierarchical']['linkage']}")

        # Train Hierarchical clustering
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.config['hierarchical']['linkage'],
            metric=self.config['hierarchical']['distance_metric']
        )

        labels = self.hierarchical_model.fit_predict(X)
        self.cluster_labels['hierarchical'] = labels

        # Calculate metrics
        metrics = self._calculate_metrics(X, labels, 'Hierarchical')
        self.cluster_metrics['hierarchical'] = metrics

        if self.verbose:
            print(f"✓ Hierarchical clustering complete")
            self._print_metrics(metrics)
            self._print_cluster_distribution(labels)

        return self.hierarchical_model, labels

    def dbscan_clustering(self, X: np.ndarray,
                         eps: float = None,
                         min_samples: int = None,
                         find_optimal_eps: bool = True) -> Tuple[DBSCAN, np.ndarray]:
        """
        Perform DBSCAN clustering (density-based)

        Args:
            X: Feature matrix (scaled)
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            find_optimal_eps: Whether to find optimal eps

        Returns:
            Tuple of (fitted model, cluster labels)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("DBSCAN CLUSTERING")
            print("="*60)

        if min_samples is None:
            min_samples = self.config['dbscan']['min_samples']

        # Find optimal eps if requested
        if find_optimal_eps and eps is None:
            if self.verbose:
                print("Finding optimal eps parameter...")
            eps = self._find_optimal_eps(X, min_samples)

        if eps is None:
            eps = 0.5  # Default

        if self.verbose:
            print(f"Training DBSCAN with eps={eps:.2f}, min_samples={min_samples}...")

        # Train DBSCAN
        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = self.dbscan_model.fit_predict(X)
        self.cluster_labels['dbscan'] = labels

        # Calculate metrics (only for non-noise points)
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 1:
            metrics = self._calculate_metrics(
                X[non_noise_mask],
                labels[non_noise_mask],
                'DBSCAN'
            )
            metrics['noise_points'] = (labels == -1).sum()
            metrics['noise_percentage'] = (labels == -1).sum() / len(labels) * 100
        else:
            metrics = {
                'algorithm': 'DBSCAN',
                'noise_points': (labels == -1).sum(),
                'noise_percentage': 100.0,
                'silhouette_score': None,
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None
            }

        self.cluster_metrics['dbscan'] = metrics

        if self.verbose:
            print(f"✓ DBSCAN clustering complete")
            self._print_metrics(metrics)
            self._print_cluster_distribution(labels)

        return self.dbscan_model, labels

    def fit_dbscan(self, X: np.ndarray,
                   eps: float = None,
                   min_samples: int = None) -> np.ndarray:
        """
        Convenience method: Fit DBSCAN and return labels only

        Args:
            X: Feature matrix (scaled)
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood

        Returns:
            Cluster labels
        """
        _, labels = self.dbscan_clustering(X, eps=eps, min_samples=min_samples, find_optimal_eps=False)
        return labels

    def _find_optimal_k(self, X: np.ndarray) -> int:
        """
        Find optimal k using elbow method

        Args:
            X: Feature matrix

        Returns:
            Optimal number of clusters
        """
        k_range = self.config['kmeans']['n_clusters_range']

        inertias = []
        silhouettes = []

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.config['kmeans']['n_init']
            )
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)

            if k > 1:
                silhouettes.append(silhouette_score(X, labels))
            else:
                silhouettes.append(0)

        # Store elbow data
        self.elbow_data = pd.DataFrame({
            'k': list(k_range),
            'inertia': inertias,
            'silhouette': silhouettes
        })

        # Find elbow using second derivative
        inertia_diff = np.diff(inertias)
        inertia_diff2 = np.diff(inertia_diff)
        elbow_idx = np.argmax(inertia_diff2) + 2  # +2 because of double diff

        optimal_k = list(k_range)[elbow_idx]

        if self.verbose:
            print(f"  Optimal k found: {optimal_k}")
            print(f"  (Elbow method based on inertia)")

        return optimal_k

    def _find_optimal_eps(self, X: np.ndarray, min_samples: int) -> float:
        """
        Find optimal eps for DBSCAN using k-distance graph

        Args:
            X: Feature matrix
            min_samples: Minimum samples parameter

        Returns:
            Optimal eps value
        """
        from sklearn.neighbors import NearestNeighbors

        # Calculate k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)

        # Sort distances
        distances = np.sort(distances[:, -1], axis=0)

        # Find elbow in k-distance graph
        # Use 75th percentile instead of median to avoid 0.0 values
        optimal_eps = np.percentile(distances, 75)

        # Ensure eps is greater than 0
        if optimal_eps <= 0:
            # Fallback 1: Try 90th percentile
            optimal_eps = np.percentile(distances, 90)

        if optimal_eps <= 0:
            # Fallback 2: Try max distance
            optimal_eps = np.max(distances)

        if optimal_eps <= 0:
            # Fallback 3: Use mean of non-zero distances
            non_zero_distances = distances[distances > 0]
            if len(non_zero_distances) > 0:
                optimal_eps = np.mean(non_zero_distances)
            else:
                # Last resort: use a small default value based on data scale
                # For PCA-reduced data, a value around 1.0-2.0 usually works
                optimal_eps = 1.5
                if self.verbose:
                    print(f"  Warning: All distances are 0, using default eps={optimal_eps}")

        if self.verbose:
            print(f"  Optimal eps found: {optimal_eps:.3f}")
            print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")

        return float(optimal_eps)

    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray,
                          algorithm_name: str) -> Dict:
        """
        Calculate clustering evaluation metrics

        Args:
            X: Feature matrix
            labels: Cluster labels
            algorithm_name: Name of clustering algorithm

        Returns:
            Dictionary of metrics
        """
        metrics = {'algorithm': algorithm_name}

        try:
            # Silhouette Score (higher is better, range [-1, 1])
            metrics['silhouette_score'] = silhouette_score(X, labels)

            # Calinski-Harabasz Score (higher is better)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)

            # Davies-Bouldin Score (lower is better)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not calculate some metrics: {e}")
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['davies_bouldin_score'] = None

        metrics['n_clusters'] = len(np.unique(labels[labels != -1]))  # Exclude noise

        return metrics

    def _print_metrics(self, metrics: Dict):
        """Print clustering metrics in a readable format"""
        print(f"\n  Clustering Metrics:")
        print(f"    Number of clusters: {metrics['n_clusters']}")

        if metrics['silhouette_score'] is not None:
            print(f"    Silhouette Score: {metrics['silhouette_score']:.4f}")
        if metrics['calinski_harabasz_score'] is not None:
            print(f"    Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
        if metrics['davies_bouldin_score'] is not None:
            print(f"    Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")

        if 'noise_points' in metrics:
            print(f"    Noise points: {metrics['noise_points']} ({metrics['noise_percentage']:.1f}%)")

    def _print_cluster_distribution(self, labels: np.ndarray):
        """Print distribution of samples across clusters"""
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  Cluster Distribution:")
        for label, count in zip(unique, counts):
            if label == -1:
                print(f"    Noise: {count} samples")
            else:
                print(f"    Cluster {label}: {count} samples")

    def cluster_all(self, X: np.ndarray,
                   kmeans_k: int = None,
                   hierarchical_k: int = None,
                   dbscan_eps: float = None) -> Dict[str, np.ndarray]:
        """
        Run all clustering algorithms

        Args:
            X: Feature matrix
            kmeans_k: Number of clusters for K-Means (None = auto)
            hierarchical_k: Number of clusters for Hierarchical (None = use config)
            dbscan_eps: Eps parameter for DBSCAN (None = auto)

        Returns:
            Dictionary of cluster labels for each algorithm
        """
        if self.verbose:
            print("\n" + "="*60)
            print("RUNNING ALL CLUSTERING ALGORITHMS")
            print("="*60)

        # K-Means
        self.kmeans_clustering(X, n_clusters=kmeans_k, find_optimal=True)

        # Hierarchical
        self.hierarchical_clustering(X, n_clusters=hierarchical_k)

        # DBSCAN
        self.dbscan_clustering(X, eps=dbscan_eps, find_optimal_eps=True)

        if self.verbose:
            print("\n" + "="*60)
            print("ALL CLUSTERING COMPLETE")
            print("="*60)

        return self.cluster_labels

    def get_cluster_profiles(self, X: np.ndarray, labels: np.ndarray,
                           feature_names: List[str],
                           df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create cluster profiles showing mean feature values per cluster

        Args:
            X: Feature matrix (scaled)
            labels: Cluster labels
            feature_names: Names of features
            df: Original DataFrame (optional, for additional context)

        Returns:
            DataFrame with cluster profiles
        """
        # Create DataFrame with features and labels
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df['cluster'] = labels

        # Calculate mean values per cluster
        cluster_profiles = feature_df.groupby('cluster').mean()

        # Calculate cluster sizes
        cluster_profiles['size'] = feature_df.groupby('cluster').size()

        return cluster_profiles

    def compare_algorithms(self) -> pd.DataFrame:
        """
        Compare all clustering algorithms based on metrics

        Returns:
            DataFrame comparing algorithms
        """
        if not self.cluster_metrics:
            raise ValueError("No clustering has been performed yet")

        comparison_data = []
        for algo_name, metrics in self.cluster_metrics.items():
            row = {
                'Algorithm': algo_name.upper(),
                'N_Clusters': metrics.get('n_clusters', 'N/A'),
                'Silhouette': metrics.get('silhouette_score', 'N/A'),
                'Calinski-Harabasz': metrics.get('calinski_harabasz_score', 'N/A'),
                'Davies-Bouldin': metrics.get('davies_bouldin_score', 'N/A')
            }

            if 'noise_percentage' in metrics:
                row['Noise_%'] = f"{metrics['noise_percentage']:.1f}%"

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)


class HyperparameterTuner:
    """
    Performs hyperparameter tuning for clustering algorithms
    """

    def __init__(self, random_state: int = RANDOM_STATE, verbose: bool = VERBOSE):
        """
        Initialize hyperparameter tuner

        Args:
            random_state: Random seed for reproducibility
            verbose: Print progress messages
        """
        self.random_state = random_state
        self.verbose = verbose
        self.tuning_results: Dict[str, pd.DataFrame] = {}
        self.best_params: Dict[str, Dict] = {}

    def tune_kmeans(self, X: np.ndarray,
                   param_grid: Dict = None,
                   metric: str = 'silhouette') -> pd.DataFrame:
        """
        Tune K-Means hyperparameters using grid search

        Args:
            X: Feature matrix
            param_grid: Dictionary of parameters to search
                       Example: {'n_clusters': [3, 4, 5, 6], 'n_init': [10, 20]}
            metric: Metric to optimize ('silhouette', 'calinski_harabasz', 'davies_bouldin')

        Returns:
            DataFrame with tuning results
        """
        if param_grid is None:
            param_grid = {
                'n_clusters': range(2, 11),
                'n_init': [10, 20, 50],
                'max_iter': [300, 500]
            }

        if self.verbose:
            print("="*60)
            print("K-MEANS HYPERPARAMETER TUNING")
            print("="*60)
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            print(f"Testing {total_combinations} parameter combinations...")

        results = []
        grid = ParameterGrid(param_grid)

        for i, params in enumerate(grid):
            if self.verbose and (i + 1) % 2 == 0:
                print(f"  Progress: {i+1}/{len(grid)} combinations tested")

            try:
                # Train model
                model = KMeans(random_state=self.random_state, **params)
                labels = model.fit_predict(X)

                # Calculate metrics
                n_clusters = len(np.unique(labels))

                result = {
                    'n_clusters': params['n_clusters'],
                    'n_init': params.get('n_init', 10),
                    'max_iter': params.get('max_iter', 300),
                    'inertia': model.inertia_
                }

                if n_clusters > 1:
                    result['silhouette'] = silhouette_score(X, labels)
                    result['calinski_harabasz'] = calinski_harabasz_score(X, labels)
                    result['davies_bouldin'] = davies_bouldin_score(X, labels)
                else:
                    result['silhouette'] = -1
                    result['calinski_harabasz'] = 0
                    result['davies_bouldin'] = 999

                results.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed for params {params}: {e}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Find best parameters
        if metric == 'silhouette':
            best_idx = results_df['silhouette'].idxmax()
        elif metric == 'calinski_harabasz':
            best_idx = results_df['calinski_harabasz'].idxmax()
        elif metric == 'davies_bouldin':
            best_idx = results_df['davies_bouldin'].idxmin()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        self.best_params['kmeans'] = results_df.loc[best_idx].to_dict()
        self.tuning_results['kmeans'] = results_df

        if self.verbose:
            print(f"\nK-Means tuning complete")
            print(f"  Best parameters (by {metric}):")
            print(f"    n_clusters: {int(self.best_params['kmeans']['n_clusters'])}")
            print(f"    n_init: {int(self.best_params['kmeans']['n_init'])}")
            print(f"    max_iter: {int(self.best_params['kmeans']['max_iter'])}")
            print(f"  Best {metric} score: {self.best_params['kmeans'][metric]:.4f}")

        return results_df

    def tune_hierarchical(self, X: np.ndarray,
                         param_grid: Dict = None,
                         metric: str = 'silhouette') -> pd.DataFrame:
        """
        Tune Hierarchical clustering hyperparameters

        Args:
            X: Feature matrix
            param_grid: Dictionary of parameters to search
            metric: Metric to optimize

        Returns:
            DataFrame with tuning results
        """
        if param_grid is None:
            param_grid = {
                'n_clusters': range(2, 11),
                'linkage': ['ward', 'complete', 'average'],
                'metric': ['euclidean']  # ward only works with euclidean
            }

        if self.verbose:
            print("\n" + "="*60)
            print("HIERARCHICAL CLUSTERING HYPERPARAMETER TUNING")
            print("="*60)
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            print(f"Testing {total_combinations} parameter combinations...")

        results = []
        grid = ParameterGrid(param_grid)

        for i, params in enumerate(grid):
            if self.verbose and (i + 1) % 2 == 0:
                print(f"  Progress: {i+1}/{len(grid)} combinations tested")

            # Skip invalid combinations
            if params['linkage'] == 'ward' and params.get('metric', 'euclidean') != 'euclidean':
                continue

            try:
                # Train model
                model_params = {k: v for k, v in params.items() if k != 'metric'}
                if params['linkage'] != 'ward':
                    model_params['metric'] = params.get('metric', 'euclidean')

                model = AgglomerativeClustering(**model_params)
                labels = model.fit_predict(X)

                # Calculate metrics
                result = {
                    'n_clusters': params['n_clusters'],
                    'linkage': params['linkage'],
                    'metric': params.get('metric', 'euclidean')
                }

                n_clusters = len(np.unique(labels))
                if n_clusters > 1:
                    result['silhouette'] = silhouette_score(X, labels)
                    result['calinski_harabasz'] = calinski_harabasz_score(X, labels)
                    result['davies_bouldin'] = davies_bouldin_score(X, labels)
                else:
                    result['silhouette'] = -1
                    result['calinski_harabasz'] = 0
                    result['davies_bouldin'] = 999

                results.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed for params {params}: {e}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Find best parameters
        if metric == 'silhouette':
            best_idx = results_df['silhouette'].idxmax()
        elif metric == 'calinski_harabasz':
            best_idx = results_df['calinski_harabasz'].idxmax()
        elif metric == 'davies_bouldin':
            best_idx = results_df['davies_bouldin'].idxmin()

        self.best_params['hierarchical'] = results_df.loc[best_idx].to_dict()
        self.tuning_results['hierarchical'] = results_df

        if self.verbose:
            print(f"\n✓ Hierarchical tuning complete")
            print(f"  Best parameters (by {metric}):")
            print(f"    n_clusters: {int(self.best_params['hierarchical']['n_clusters'])}")
            print(f"    linkage: {self.best_params['hierarchical']['linkage']}")
            print(f"  Best {metric} score: {self.best_params['hierarchical'][metric]:.4f}")

        return results_df

    def tune_dbscan(self, X: np.ndarray,
                   param_grid: Dict = None,
                   metric: str = 'silhouette',
                   min_cluster_size: int = 10) -> pd.DataFrame:
        """
        Tune DBSCAN hyperparameters

        Args:
            X: Feature matrix
            param_grid: Dictionary of parameters to search
            metric: Metric to optimize
            min_cluster_size: Minimum valid cluster size

        Returns:
            DataFrame with tuning results
        """
        if param_grid is None:
            # Create eps range based on data
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors.fit(X)
            distances, _ = neighbors.kneighbors(X)
            k_dist = np.sort(distances[:, -1])

            eps_min = np.percentile(k_dist, 10)
            eps_max = np.percentile(k_dist, 90)

            # Ensure eps_min is greater than 0
            if eps_min <= 0:
                # Use non-zero minimum or a small default
                non_zero_dists = k_dist[k_dist > 0]
                if len(non_zero_dists) > 0:
                    eps_min = np.min(non_zero_dists)
                else:
                    eps_min = 0.01  # Small default value

            # Ensure eps_max is greater than eps_min
            if eps_max <= eps_min:
                eps_max = eps_min * 10  # Make max 10x the min

            param_grid = {
                'eps': np.linspace(eps_min, eps_max, 10),
                'min_samples': [3, 5, 10, 15, 20]
            }

        # Validate and filter eps values - must be > 0 for DBSCAN
        if 'eps' in param_grid:
            eps_values = np.array(param_grid['eps'])
            valid_eps = eps_values[eps_values > 0]
            if len(valid_eps) == 0:
                # All eps values invalid, use defaults
                valid_eps = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
            param_grid['eps'] = valid_eps

        if self.verbose:
            print("\n" + "="*60)
            print("DBSCAN HYPERPARAMETER TUNING")
            print("="*60)
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            print(f"Testing {total_combinations} parameter combinations...")

        results = []
        grid = ParameterGrid(param_grid)

        for i, params in enumerate(grid):
            if self.verbose and (i + 1) % 2 == 0:
                print(f"  Progress: {i+1}/{len(grid)} combinations tested")

            try:
                # Train model
                model = DBSCAN(**params)
                labels = model.fit_predict(X)

                # Calculate metrics
                non_noise = labels != -1
                n_clusters = len(np.unique(labels[non_noise]))
                noise_count = (labels == -1).sum()
                noise_pct = noise_count / len(labels) * 100

                result = {
                    'eps': params['eps'],
                    'min_samples': params['min_samples'],
                    'n_clusters': n_clusters,
                    'noise_points': noise_count,
                    'noise_pct': noise_pct
                }

                # Only calculate metrics if we have valid clusters
                if n_clusters > 1 and non_noise.sum() > min_cluster_size:
                    result['silhouette'] = silhouette_score(X[non_noise], labels[non_noise])
                    result['calinski_harabasz'] = calinski_harabasz_score(X[non_noise], labels[non_noise])
                    result['davies_bouldin'] = davies_bouldin_score(X[non_noise], labels[non_noise])
                else:
                    result['silhouette'] = -1
                    result['calinski_harabasz'] = 0
                    result['davies_bouldin'] = 999

                results.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed for params {params}: {e}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Find best parameters (exclude configurations with too much noise)
        valid_results = results_df[results_df['noise_pct'] < 50]

        if len(valid_results) > 0:
            if metric == 'silhouette':
                best_idx = valid_results['silhouette'].idxmax()
            elif metric == 'calinski_harabasz':
                best_idx = valid_results['calinski_harabasz'].idxmax()
            elif metric == 'davies_bouldin':
                best_idx = valid_results['davies_bouldin'].idxmin()

            self.best_params['dbscan'] = results_df.loc[best_idx].to_dict()

            if self.verbose:
                print(f"\n✓ DBSCAN tuning complete")
                print(f"  Best parameters (by {metric}):")
                print(f"    eps: {self.best_params['dbscan']['eps']:.4f}")
                print(f"    min_samples: {int(self.best_params['dbscan']['min_samples'])}")
                print(f"    n_clusters: {int(self.best_params['dbscan']['n_clusters'])}")
                print(f"    noise_pct: {self.best_params['dbscan']['noise_pct']:.1f}%")
                print(f"  Best {metric} score: {self.best_params['dbscan'][metric]:.4f}")
        else:
            if self.verbose:
                print("\n  Warning: No valid parameter combinations found (all had >50% noise)")

        self.tuning_results['dbscan'] = results_df

        return results_df

    def get_best_params(self, algorithm: str = None) -> Dict:
        """
        Get best parameters for an algorithm

        Args:
            algorithm: Algorithm name ('kmeans', 'hierarchical', 'dbscan')
                      If None, returns all best parameters

        Returns:
            Dictionary of best parameters
        """
        if algorithm is None:
            return self.best_params

        if algorithm not in self.best_params:
            raise ValueError(f"No tuning results for {algorithm}. Run tune_{algorithm} first.")

        return self.best_params[algorithm]

    def get_tuning_results(self, algorithm: str = None) -> pd.DataFrame:
        """
        Get full tuning results

        Args:
            algorithm: Algorithm name ('kmeans', 'hierarchical', 'dbscan')
                      If None, returns all results

        Returns:
            DataFrame with tuning results
        """
        if algorithm is None:
            return self.tuning_results

        if algorithm not in self.tuning_results:
            raise ValueError(f"No tuning results for {algorithm}. Run tune_{algorithm} first.")

        return self.tuning_results[algorithm]


if __name__ == "__main__":
    # Example usage
    print("Clustering Models Module - Example Usage\n")

    # Generate sample data
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=100, n_features=5, centers=4, random_state=42)

    # Initialize clusterer
    clusterer = BrandClusterer()

    # Run all algorithms
    all_labels = clusterer.cluster_all(X)

    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    print(clusterer.compare_algorithms())

    print("\n" + "="*60)
    print("ELBOW DATA (K-Means)")
    print("="*60)
    if clusterer.elbow_data is not None:
        print(clusterer.elbow_data)
