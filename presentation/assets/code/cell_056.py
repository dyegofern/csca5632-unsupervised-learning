# Visualize hierarchical clusters (PRIMARY MODEL)
# Focus on this visualization for business interpretation

fig = viz.plot_clusters_2d(
    X_tsne,
    hierarchical_labels,
    title="Hierarchical Clustering - 5 Brand Segments (Primary Model)",
    brand_names=full_df['brand_name'].tolist(),
    show_labels=False
)
plt.show()

# Plot cluster sizes
fig = viz.plot_cluster_sizes(
    hierarchical_labels,
    title="Hierarchical Clustering: Segment Distribution"
)
plt.show()

print(f"\n{'='*80}")
print("HIERARCHICAL CLUSTERING RESULTS (PRIMARY MODEL)")
print(f"{'='*80}")
print(f"Number of clusters: {len(set(hierarchical_labels))}")
print(f"{'='*80}")
