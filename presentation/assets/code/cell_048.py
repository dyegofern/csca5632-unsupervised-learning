# Visualize DBSCAN tuning results
fig = viz.plot_hyperparameter_tuning(
    dbscan_tuning_results,
    algorithm='dbscan'
)
plt.show()

# Get best parameters (if available)
if 'dbscan' in tuner.best_params:
    best_dbscan_params = tuner.get_best_params('dbscan')
    print(f"\nBest DBSCAN parameters:")
    print(f"  eps: {best_dbscan_params['eps']:.4f}")
    print(f"  min_samples: {int(best_dbscan_params['min_samples'])}")
    print(f"  n_clusters: {int(best_dbscan_params['n_clusters'])}")
    print(f"  noise_pct: {best_dbscan_params['noise_pct']:.1f}%")
    print(f"  Silhouette score: {best_dbscan_params['silhouette']:.4f}")
else:
    print("\nNo valid DBSCAN parameters found")