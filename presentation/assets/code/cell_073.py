# Visualize Top 5 Industries per Cluster
if 'industry_name' in full_df.columns and 'hierarchical_cluster' in full_df.columns:
    n_clusters = full_df['hierarchical_cluster'].nunique()
    
    # Create subplots for each cluster
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5))
    if n_clusters == 1:
        axes = [axes]
    
    for idx, cluster_id in enumerate(sorted(full_df['hierarchical_cluster'].unique())):
        cluster_data = full_df[full_df['hierarchical_cluster'] == cluster_id]
        top_industries = cluster_data['industry_name'].value_counts().head(5)
        
        ax = axes[idx]
        top_industries.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(f'Cluster {cluster_id}\n(n={len(cluster_data)})', fontweight='bold')
        ax.set_xlabel('Count')
        ax.set_ylabel('')
        ax.invert_yaxis()
        
        # Add count labels on bars
        for i, v in enumerate(top_industries.values):
            ax.text(v + 0.5, i, str(v), va='center')
    
    plt.suptitle('Top 5 Industries per Cluster', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("Required columns not found")