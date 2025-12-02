# Analyze how industries are distributed across clusters
if 'industry_name' in full_df.columns and 'hierarchical_cluster' in full_df.columns:
    # Show top 3 industries per cluster
    print("Top 3 Industries per Cluster:")
    print("="*60)
    for cluster_id in sorted(full_df['hierarchical_cluster'].unique()):
        cluster_data = full_df[full_df['hierarchical_cluster'] == cluster_id]
        top_industries = cluster_data['industry_name'].value_counts().head(3)
        print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
        for industry, count in top_industries.items():
            pct = (count / len(cluster_data)) * 100
            print(f"  {industry}: {count} ({pct:.1f}%)")
    
    # Visualize top 5 industries across clusters
    print("\n" + "="*60)
    top_industries = full_df['industry_name'].value_counts().head(5).index
    industry_subset = full_df[full_df['industry_name'].isin(top_industries)]
    
    industry_cluster_pct = pd.crosstab(
        industry_subset['industry_name'],
        industry_subset['hierarchical_cluster'],
        normalize='index'
    ) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    industry_cluster_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    ax.set_xlabel('Industry', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Top 5 Industries Distribution Across Clusters', fontsize=14, fontweight='bold')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("Required columns not found")