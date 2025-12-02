    if 'kmeans' in tuner.best_params:
        kmeans_best = tuner.get_best_params('kmeans')
        print("\nK-Means:")
        print(f"  n_clusters: {int(kmeans_best['n_clusters'])}")
        print(f"  n_init: {int(kmeans_best['n_init'])}")
        print(f"  Silhouette Score: {kmeans_best['silhouette']:.4f}")
    
    if 'hierarchical' in tuner.best_params:
        hier_best = tuner.get_best_params('hierarchical')
        print("\nHierarchical Clustering:")
        print(f"  n_clusters: {int(hier_best['n_clusters'])}")
        print(f"  linkage: {hier_best['linkage']}")
        print(f"  Silhouette Score: {hier_best['silhouette']:.4f}")
    
    if 'dbscan' in tuner.best_params:
        dbscan_best = tuner.get_best_params('dbscan')
        print("\nDBSCAN:")
        print(f"  eps: {dbscan_best['eps']:.4f}")
        print(f"  min_samples: {int(dbscan_best['min_samples'])}")
        print(f"  n_clusters: {int(dbscan_best['n_clusters'])}")
        print(f"  noise_pct: {dbscan_best['noise_pct']:.1f}%")
        print(f"  Silhouette Score: {dbscan_best['silhouette']:.4f}")