
kmeans_tuning_results = tuner.tune_kmeans(
    X_pca,
    param_grid=kmeans_param_grid,
    metric=OPTIMIZATION_METRIC
)
fig = viz.plot_hyperparameter_tuning(
    kmeans_tuning_results,
    algorithm='kmeans',
    metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
)
plt.show()