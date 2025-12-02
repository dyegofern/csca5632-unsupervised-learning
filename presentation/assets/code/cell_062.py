# Visualize DBSCAN clusters
fig = viz.plot_clusters_2d(
    X_tsne,
    dbscan_labels,
    title="DBSCAN Clustering (t-SNE Visualization)",
    show_legend=True  # Hide legend if there are too many clusters
)
plt.show()

# Examine outliers (noise points)
outliers = full_df[full_df['dbscan_cluster'] == -1]
print(f"\nDBSCAN identified {len(outliers)} outlier brands:")
if len(outliers) > 0:
    display_cols = ['brand_name', 'company_name', 'industry_name']
    if 'initial_greenwashing_level' in outliers.columns:
        display_cols.append('initial_greenwashing_level')
    available_cols = [c for c in display_cols if c in outliers.columns]
    print(outliers[available_cols].head(10))