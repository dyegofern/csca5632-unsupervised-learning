# Visualize K-Means tuning results
fig = viz.plot_hyperparameter_tuning(
    kmeans_tuning_results,
    algorithm='kmeans',
    metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
)
plt.show()

# Get best parameters
best_kmeans_params = tuner.get_best_params('kmeans')
print(f"\nBest K-Means parameters:")
print(f"  n_clusters: {int(best_kmeans_params['n_clusters'])}")
print(f"  n_init: {int(best_kmeans_params['n_init'])}")
print(f"  Silhouette score: {best_kmeans_params['silhouette']:.4f}")