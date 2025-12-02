# K-Means visualization (for reference only)

fig = viz.plot_cluster_sizes(kmeans_labels, title="K-Means Cluster Distribution (Reference Only)")
plt.show()

print(f"\nK-Means: {len(set(kmeans_labels))} clusters")
print("Note: K-Means had lower performance (silhouette=0.54) compared to other methods.")
print("See Hierarchical and DBSCAN visualizations above for primary analysis.")
