hierarchical_tuning_results = tuner.tune_hierarchical(
    X_pca,
    param_grid=hierarchical_param_grid,
    metric=OPTIMIZATION_METRIC
)
fig = viz.plot_hyperparameter_tuning(
    hierarchical_tuning_results,
    algorithm='hierarchical',
    metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin']
)
plt.show()