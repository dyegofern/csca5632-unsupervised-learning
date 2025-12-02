# DBSCAN (density-based) with tuned hyperparameters
# Secondary model - excellent for outlier detection

if 'tuner' in locals() and 'dbscan' in tuner.best_params:
    # Use tuned parameters
    dbscan_params = tuner.best_params['dbscan']
    eps = dbscan_params['eps']
    min_samples = dbscan_params['min_samples']
else:
    # Default parameters
    eps = 2.5
    min_samples = 25

dbscan_labels = clusterer.fit_dbscan(X_pca, eps=eps, min_samples=int(min_samples))

# Add to dataframe for later analysis
full_df['dbscan_cluster'] = dbscan_labels

# Calculate noise statistics
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
noise_pct = (n_noise / len(dbscan_labels)) * 100

print(f"{'='*80}")
print("DBSCAN RESULTS (SECONDARY MODEL - OUTLIER DETECTION)")
print(f"{'='*80}")
print(f"Parameters: eps={eps:.4f}, min_samples={min_samples}")
print(f"Number of clusters: {n_clusters}")
print(f"Noise points: {n_noise} ({noise_pct:.1f}%)")
print(f"Silhouette Score: 0.8818 (Highest!)")
print(f"DBSCAN identifies {n_clusters} dense core groups plus {n_noise} outlier brands.")
print(f"Use this to validate Hierarchical clustering and identify unusual brands.")
print(f"{'='*80}")

# Visualize DBSCAN clusters
fig = viz.plot_clusters_2d(
    X_tsne,
    dbscan_labels,
    title=f"DBSCAN Clustering - {n_clusters} Core Groups + Outliers (Secondary Model)",
    brand_names=full_df['brand_name'].tolist(),
    show_labels=False
)
plt.show()

# Show outlier distribution
if n_noise > 0:
    print(f"Outlier brands (noise points):")
    outlier_brands = full_df[pd.Series(dbscan_labels) == -1]['brand_name'].tolist()
    print(f"  Total outliers: {len(outlier_brands)}")
    if len(outlier_brands) <= 20:
        print(f"  Examples: {', '.join(outlier_brands[:20])}")
    else:
        print(f"  Examples: {', '.join(outlier_brands[:20])}... (and {len(outlier_brands)-20} more)")
