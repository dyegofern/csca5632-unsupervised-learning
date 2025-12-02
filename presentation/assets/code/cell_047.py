dbscan_tuning_results = tuner.tune_dbscan(
    X_tsne,
    param_grid=dbscan_param_grid,
    metric=OPTIMIZATION_METRIC,
    min_cluster_size=10
)
fig = viz.plot_hyperparameter_tuning(
    dbscan_tuning_results,
    algorithm='dbscan'
)
plt.show()