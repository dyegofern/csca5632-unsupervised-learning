best_params = tuner.get_best_params('hierarchical')
n_clusters = int(best_params['n_clusters'])
print(f"Using tuned Hierarchical parameters: n_clusters={n_clusters}, linkage={best_params['linkage']}")

# Update config for hierarchical clustering to use tuned linkage
clusterer.config['hierarchical']['linkage'] = best_params['linkage']
clusterer.config['hierarchical']['n_clusters'] = n_clusters

hierarchical_model, hierarchical_labels = clusterer.hierarchical_clustering(
    X_pca,
    n_clusters=n_clusters
)