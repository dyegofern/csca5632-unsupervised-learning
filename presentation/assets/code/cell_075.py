# Visualize Top 5 Brands per Cluster (by emissions or revenue)
if 'brand_name' in full_df.columns and 'hierarchical_cluster' in full_df.columns:
    n_clusters = full_df['hierarchical_cluster'].nunique()
    
    # Use scope12_total if available, otherwise just show top brands by count
    metric_col = 'scope12_total' if 'scope12_total' in full_df.columns else None
    metric_name = 'Emissions (Scope 1+2)' if metric_col else 'Brand Count'
    
    # Create subplots for each cluster
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5))
    if n_clusters == 1:
        axes = [axes]
    
    for idx, cluster_id in enumerate(sorted(full_df['hierarchical_cluster'].unique())):
        cluster_data = full_df[full_df['hierarchical_cluster'] == cluster_id]
        
        if metric_col:
            # Sort by metric and get top 5
            top_brands = cluster_data.nlargest(5, metric_col)[['brand_name', metric_col]].set_index('brand_name')
            values = top_brands[metric_col]
        else:
            # Just count brand occurrences
            values = cluster_data['brand_name'].value_counts().head(5)
        
        ax = axes[idx]
        values.plot(kind='barh', ax=ax, color='forestgreen')
        ax.set_title(f'Cluster {cluster_id}\n(n={len(cluster_data)})', fontweight='bold')
        ax.set_xlabel(metric_name)
        ax.set_ylabel('')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(values.values):
            if metric_col:
                label = f'{v:,.0f}' if v >= 1000 else f'{v:.0f}'
            else:
                label = str(int(v))
            ax.text(v + (v * 0.01), i, label, va='center', fontsize=9)
    
    plt.suptitle(f'Top 5 Brands per Cluster (by {metric_name})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("Required columns not found")