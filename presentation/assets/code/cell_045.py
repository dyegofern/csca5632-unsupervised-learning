# Visualize Hierarchical tuning results
fig = viz.plot_hyperparameter_tuning(
    hierarchical_tuning_results,
    algorithm='hierarchical',
    metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin']
)
plt.show()

# Get best parameters
best_hier_params = tuner.get_best_params('hierarchical')
print(f"\nBest Hierarchical parameters:")
print(f"  n_clusters: {int(best_hier_params['n_clusters'])}")
print(f"  linkage: {best_hier_params['linkage']}")
print(f"  Silhouette score: {best_hier_params['silhouette']:.4f}")