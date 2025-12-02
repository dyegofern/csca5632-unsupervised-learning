"""
Visualization Utilities Module
Creates visualizations for clustering results, dimensionality reduction, and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import VIZ_CONFIG, VERBOSE


class ClusterVisualizer:
    """
    Creates comprehensive visualizations for unsupervised learning results
    """

    def __init__(self, config: dict = VIZ_CONFIG, verbose: bool = VERBOSE):
        """
        Initialize visualizer

        Args:
            config: Visualization configuration
            verbose: Print progress messages
        """
        self.config = config
        self.verbose = verbose

        # Set style
        sns.set_style(config['style'])
        plt.rcParams['figure.figsize'] = config['figsize']
        plt.rcParams['figure.dpi'] = config['dpi']

    def plot_clusters_2d(self, X_2d: np.ndarray, labels: np.ndarray,
                        title: str = "Cluster Visualization",
                        brand_names: List[str] = None,
                        show_labels: bool = False,
                        show_legend: bool = True) -> plt.Figure:
        """
        Plot clusters in 2D space (e.g., from t-SNE or PCA)

        Args:
            X_2d: 2D feature matrix
            labels: Cluster labels
            title: Plot title
            brand_names: Optional brand names for annotation
            show_labels: Whether to show brand name labels
            show_legend: Whether to show cluster legend

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.config['figsize'])

        # Create color map
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(self.config['color_palette'], len(unique_labels))

        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                # Noise points (for DBSCAN)
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                          c='gray', marker='x', s=50, alpha=0.5,
                          label='Noise')
            else:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                          c=[colors(i)], s=100, alpha=0.6,
                          label=f'Cluster {label}', edgecolors='black', linewidth=0.5)

        # Add brand name labels if requested
        if show_labels and brand_names is not None:
            for i, name in enumerate(brand_names):
                ax.annotate(name, (X_2d[i, 0], X_2d[i, 1]),
                           fontsize=8, alpha=0.7)

        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        if show_legend:
            ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.verbose:
            print(f"2D cluster plot created: {title}")

        return fig

    def plot_clusters_3d(self, X_3d: np.ndarray, labels: np.ndarray,
                        title: str = "3D Cluster Visualization") -> plt.Figure:
        """
        Plot clusters in 3D space

        Args:
            X_3d: 3D feature matrix
            labels: Cluster labels
            title: Plot title

        Returns:
            Figure object
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=self.config['figsize'])
        ax = fig.add_subplot(111, projection='3d')

        # Create color map
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(self.config['color_palette'], len(unique_labels))

        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                          c='gray', marker='x', s=50, alpha=0.5,
                          label='Noise')
            else:
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                          c=[colors(i)], s=100, alpha=0.6,
                          label=f'Cluster {label}')

        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.set_zlabel('Dimension 3', fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')

        plt.tight_layout()

        if self.verbose:
            print(f"3D cluster plot created: {title}")

        return fig

    def plot_elbow_curve(self, elbow_data: pd.DataFrame) -> plt.Figure:
        """
        Plot elbow curve for K-Means

        Args:
            elbow_data: DataFrame with 'k', 'inertia', 'silhouette' columns

        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Inertia plot
        ax1.plot(elbow_data['k'], elbow_data['inertia'],
                marker='o', markersize=8, linewidth=2, color='steelblue')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        ax1.set_title('Elbow Method: Inertia', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Silhouette plot
        ax2.plot(elbow_data['k'], elbow_data['silhouette'],
                marker='s', markersize=8, linewidth=2, color='coral')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Elbow Method: Silhouette Score', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.verbose:
            print("Elbow curve plot created")

        return fig

    def plot_dendrogram(self, X: np.ndarray, method: str = 'ward',
                       title: str = "Hierarchical Clustering Dendrogram") -> plt.Figure:
        """
        Plot dendrogram for hierarchical clustering

        Args:
            X: Feature matrix
            method: Linkage method ('ward', 'complete', 'average', 'single')
            title: Plot title

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.config['figsize'])

        # Calculate linkage
        linkage_matrix = linkage(X, method=method)

        # Plot dendrogram
        dendrogram(linkage_matrix, ax=ax, color_threshold=0.7*max(linkage_matrix[:,2]))

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if self.verbose:
            print("Dendrogram created")

        return fig

    def plot_variance_explained(self, variance_df: pd.DataFrame,
                               n_components: int = 10) -> plt.Figure:
        """
        Plot variance explained by principal components

        Args:
            variance_df: DataFrame with variance information
            n_components: Number of components to display

        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Limit to n_components
        plot_df = variance_df.head(n_components)

        # Individual variance
        ax1.bar(range(len(plot_df)), plot_df['Variance_Explained'],
               color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Variance Explained', fontsize=12)
        ax1.set_title('Variance Explained per Component', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(plot_df)))
        ax1.set_xticklabels(plot_df['Component'], rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Cumulative variance
        ax2.plot(range(len(plot_df)), plot_df['Cumulative_Variance'],
                marker='o', markersize=8, linewidth=2, color='coral')
        ax2.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Principal Component', fontsize=12)
        ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(plot_df)))
        ax2.set_xticklabels(plot_df['Component'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.verbose:
            print("Variance explained plot created")

        return fig

    def plot_feature_importance(self, top_features_df: pd.DataFrame,
                               component: str = 'PC1',
                               n_features: int = 10) -> plt.Figure:
        """
        Plot top features for a principal component

        Args:
            top_features_df: DataFrame with top features per component
            component: Which component to plot
            n_features: Number of features to show

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter for specific component
        component_df = top_features_df[top_features_df['Component'] == component].head(n_features)

        # Sort by absolute loading
        component_df = component_df.sort_values('Abs_Loading', ascending=True)

        # Create bar plot
        colors = ['red' if x < 0 else 'green' for x in component_df['Loading']]
        ax.barh(range(len(component_df)), component_df['Loading'],
               color=colors, alpha=0.7, edgecolor='black')

        ax.set_yticks(range(len(component_df)))
        ax.set_yticklabels(component_df['Feature'])
        ax.set_xlabel('Loading', fontsize=12)
        ax.set_title(f'Top Features for {component}', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if self.verbose:
            print(f"✓ Feature importance plot created for {component}")

        return fig

    def plot_cluster_profiles(self, cluster_profiles: pd.DataFrame,
                             top_n_features: int = 10) -> plt.Figure:
        """
        Plot heatmap of cluster profiles

        Args:
            cluster_profiles: DataFrame with mean feature values per cluster
            top_n_features: Number of features to display

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Select top features by variance across clusters
        feature_variance = cluster_profiles.iloc[:, :-1].var()  # Exclude 'size' column
        top_features = feature_variance.nlargest(top_n_features).index

        # Create heatmap
        plot_data = cluster_profiles[top_features].T
        sns.heatmap(plot_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   center=0, ax=ax, cbar_kws={'label': 'Feature Value (scaled)'})

        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Cluster Profiles (Mean Feature Values)', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if self.verbose:
            print("✓ Cluster profiles heatmap created")

        return fig

    def plot_cluster_sizes(self, labels: np.ndarray, title: str = "Cluster Sizes") -> plt.Figure:
        """
        Plot distribution of samples across clusters

        Args:
            labels: Cluster labels
            title: Plot title

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Count samples per cluster
        unique, counts = np.unique(labels, return_counts=True)

        # Create labels
        cluster_labels = []
        for label in unique:
            if label == -1:
                cluster_labels.append('Noise')
            else:
                cluster_labels.append(f'Cluster {label}')

        # Create bar plot
        colors = ['gray' if label == -1 else plt.cm.get_cmap(self.config['color_palette'])(i/len(unique))
                 for i, label in enumerate(unique)]

        ax.bar(cluster_labels, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add count labels on bars
        for i, (label, count) in enumerate(zip(cluster_labels, counts)):
            ax.text(i, count + max(counts)*0.02, str(count),
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if self.verbose:
            print("✓ Cluster sizes plot created")

        return fig

    def plot_comparison_scatter(self, X_2d: np.ndarray,
                               labels_dict: dict,
                               brand_names: List[str] = None) -> plt.Figure:
        """
        Compare multiple clustering algorithms side-by-side

        Args:
            X_2d: 2D feature matrix
            labels_dict: Dictionary with algorithm names as keys, labels as values
            brand_names: Optional brand names

        Returns:
            Figure object
        """
        n_algorithms = len(labels_dict)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 5))

        if n_algorithms == 1:
            axes = [axes]

        for ax, (algo_name, labels) in zip(axes, labels_dict.items()):
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap(self.config['color_palette'], len(unique_labels))
                
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if label == -1:
                    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                             c='gray', marker='x', s=50, alpha=0.5,
                             label='Noise')
                else:
                    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                             c=[colors(i)], s=100, alpha=0.6,
                             label=f'Cluster {label}', edgecolors='black', linewidth=0.5)

            ax.set_xlabel('Dimension 1', fontsize=10)
            ax.set_ylabel('Dimension 2', fontsize=10)
            ax.set_title(algo_name.upper(), fontsize=12, fontweight='bold')
            if (algo_name != 'DBSCAN'):
                ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Clustering Algorithm Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if self.verbose:
            print(f"Comparison scatter plot created for {n_algorithms} algorithms")

        return fig

    def plot_hyperparameter_tuning(self, tuning_results: pd.DataFrame,
                                   algorithm: str = 'K-Means',
                                   metrics: List[str] = None) -> plt.Figure:
        """
        Visualize hyperparameter tuning results

        Args:
            tuning_results: DataFrame with tuning results
            algorithm: Algorithm name ('K-Means', 'Hierarchical', 'DBSCAN')
            metrics: List of metrics to plot (default: all available)

        Returns:
            Figure object
        """
        if metrics is None:
            available_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
            metrics = [m for m in available_metrics if m in tuning_results.columns]

        if algorithm.lower() == 'kmeans':
            return self._plot_kmeans_tuning(tuning_results, metrics)
        elif algorithm.lower() == 'hierarchical':
            return self._plot_hierarchical_tuning(tuning_results, metrics)
        elif algorithm.lower() == 'dbscan':
            return self._plot_dbscan_tuning(tuning_results, metrics)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _plot_kmeans_tuning(self, results: pd.DataFrame, metrics: List[str]) -> plt.Figure:
        """Plot K-Means hyperparameter tuning results"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            # Group by n_clusters and plot
            grouped = results.groupby('n_clusters')[metric].mean()

            ax.plot(grouped.index, grouped.values, marker='o', linewidth=2,
                   markersize=8, label=metric)
            ax.fill_between(grouped.index,
                           results.groupby('n_clusters')[metric].min(),
                           results.groupby('n_clusters')[metric].max(),
                           alpha=0.2)

            # Mark best value
            if metric == 'davies_bouldin':
                best_k = grouped.idxmin()
            else:
                best_k = grouped.idxmax()

            ax.scatter([best_k], [grouped[best_k]], color='red',
                      s=200, zorder=5, marker='*',
                      label=f'Best k={best_k}')

            ax.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} vs k', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('K-Means Hyperparameter Tuning Results', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if self.verbose:
            print("K-Means tuning visualization created")

        return fig

    def _plot_hierarchical_tuning(self, results: pd.DataFrame, metrics: List[str]) -> plt.Figure:
        """Plot Hierarchical clustering hyperparameter tuning results"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        linkage_types = results['linkage'].unique()
        colors = plt.cm.get_cmap('tab10', len(linkage_types))

        for ax, metric in zip(axes, metrics):
            for i, linkage in enumerate(linkage_types):
                linkage_data = results[results['linkage'] == linkage]
                grouped = linkage_data.groupby('n_clusters')[metric].mean()

                ax.plot(grouped.index, grouped.values, marker='o', linewidth=2,
                       markersize=8, label=linkage, color=colors(i))

            ax.set_xlabel('Number of Clusters', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Clusters', fontsize=12, fontweight='bold')
            ax.legend(title='Linkage')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Hierarchical Clustering Hyperparameter Tuning', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if self.verbose:
            print("Hierarchical tuning visualization created")

        return fig

    def _plot_dbscan_tuning(self, results: pd.DataFrame, metrics: List[str]) -> plt.Figure:
        """Plot DBSCAN hyperparameter tuning results"""
        # Create 2x2 grid: heatmap + 3 metric plots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Format eps values for display (round to avoid huge numbers)
        eps_values = sorted(results['eps'].unique())
        eps_labels = [f'{eps:.3f}' if eps < 1 else f'{eps:.2f}' if eps < 10 else f'{eps:.1f}' for eps in eps_values]

        # Silhouette score heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        pivot_data = results.pivot_table(
            values='silhouette',
            index='min_samples',
            columns='eps',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=False, fmt='.3f', cmap='RdYlGn',
                   ax=ax1, cbar_kws={'label': 'Silhouette Score'})
        ax1.set_title('Silhouette Score Heatmap', fontsize=12, fontweight='bold')
        ax1.set_xlabel('eps', fontsize=11)
        ax1.set_ylabel('min_samples', fontsize=11)
        # Set custom x-axis labels with fewer ticks
        n_ticks = min(10, len(eps_labels))
        tick_indices = np.linspace(0, len(eps_labels)-1, n_ticks, dtype=int)
        ax1.set_xticks([i + 0.5 for i in tick_indices])
        ax1.set_xticklabels([eps_labels[i] for i in tick_indices], rotation=45, ha='right')

        # Number of clusters heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        pivot_clusters = results.pivot_table(
            values='n_clusters',
            index='min_samples',
            columns='eps',
            aggfunc='mean'
        )
        sns.heatmap(pivot_clusters, annot=True, fmt='.0f', cmap='viridis',
                   ax=ax2, cbar_kws={'label': 'Number of Clusters'})
        ax2.set_title('Number of Clusters', fontsize=12, fontweight='bold')
        ax2.set_xlabel('eps', fontsize=11)
        ax2.set_ylabel('min_samples', fontsize=11)
        # Set custom x-axis labels
        ax2.set_xticks([i + 0.5 for i in tick_indices])
        ax2.set_xticklabels([eps_labels[i] for i in tick_indices], rotation=45, ha='right')

        # Noise percentage heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        pivot_noise = results.pivot_table(
            values='noise_pct',
            index='min_samples',
            columns='eps',
            aggfunc='mean'
        )
        sns.heatmap(pivot_noise, annot=False, fmt='.1f', cmap='RdYlGn_r',
                   ax=ax3, cbar_kws={'label': 'Noise %'})
        ax3.set_title('Noise Percentage', fontsize=12, fontweight='bold')
        ax3.set_xlabel('eps', fontsize=11)
        ax3.set_ylabel('min_samples', fontsize=11)
        # Set custom x-axis labels
        ax3.set_xticks([i + 0.5 for i in tick_indices])
        ax3.set_xticklabels([eps_labels[i] for i in tick_indices], rotation=45, ha='right')

        # Parameter space scatter plot
        ax4 = fig.add_subplot(gs[1, 1])
        scatter = ax4.scatter(results['eps'], results['min_samples'],
                            c=results['silhouette'], s=100,
                            cmap='RdYlGn', alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, ax=ax4, label='Silhouette Score')
        ax4.set_xlabel('eps', fontsize=11)
        ax4.set_ylabel('min_samples', fontsize=11)
        ax4.set_title('Parameter Space (colored by Silhouette)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        # Format x-axis for scatter plot
        ax4.ticklabel_format(axis='x', style='scientific', scilimits=(-2, 3))

        plt.suptitle('DBSCAN Hyperparameter Tuning Results', fontsize=14, fontweight='bold')

        if self.verbose:
            print("DBSCAN tuning visualization created")
        return fig


if __name__ == "__main__":
    # Example usage
    print("Visualization Utilities Module - Example Usage\n")

    # Generate sample data
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=100, n_features=2, centers=4, random_state=42)

    # Initialize visualizer
    viz = ClusterVisualizer()

    # Create sample labels
    labels = y

    # Plot 2D clusters
    fig = viz.plot_clusters_2d(X, labels, title="Sample Cluster Visualization")
    plt.show()

    # Plot cluster sizes
    fig = viz.plot_cluster_sizes(labels, title="Sample Cluster Distribution")
    plt.show()

    print("\n" + "="*60)
    print("VISUALIZATION MODULE READY")
    print("="*60)
    print("Available visualizations:")
    print("  - 2D/3D cluster plots")
    print("  - Elbow curves")
    print("  - Dendrograms")
    print("  - Variance explained")
    print("  - Feature importance")
    print("  - Cluster profiles heatmaps")
    print("  - Algorithm comparisons")
