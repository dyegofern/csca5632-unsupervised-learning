
best_params = tuner.get_best_params('kmeans')
n_clusters = int(best_params['n_clusters'])
print(f"Using tuned K-Means parameters: n_clusters={n_clusters}")
kmeans_model, kmeans_labels = clusterer.kmeans_clustering(
    X_pca,
    n_clusters=n_clusters,
    find_optimal=False  # Use specified k from tuning
)