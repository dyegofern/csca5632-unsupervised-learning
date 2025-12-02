# Create cluster labels with standardized format for Hierarchical clustering

n_clusters = len(full_df['hierarchical_cluster'].unique())
cluster_labels = {i: f'CL-{i+1:02d}' for i in range(n_clusters)}

# Map to dataframe
full_df['segment_label'] = full_df['hierarchical_cluster'].map(cluster_labels)

print("\nStandardized Segment Labels (Hierarchical Clustering):")
print("=" * 60)
for cluster_id in sorted(cluster_labels.keys()):
    label = cluster_labels[cluster_id]
    count = (full_df['hierarchical_cluster'] == cluster_id).sum()
    pct = count / len(full_df) * 100
    print(f"  {label}: Cluster {cluster_id} ({count} brands, {pct:.1f}%)")

print("\nSegment Label Distribution:")
print(full_df['segment_label'].value_counts().sort_index())

print("\n" + "="*60)
print("Note: Focus business interpretation on these 5 segments.")
print("DBSCAN results available for outlier validation.")
print("="*60)
