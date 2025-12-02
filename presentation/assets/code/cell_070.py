# Analyze the relationship between brand clusters and company-level greenwashing risk
if 'environmental_risk_score' in full_df.columns and 'segment_label' in full_df.columns:
    # Create binned categories for environmental risk score
    max_risk = full_df['environmental_risk_score'].max()
    bins = np.linspace(0, max_risk, 11)  # 11 edges = 10 bins
    labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    
    # Create binned version of environmental risk score
    full_df['env_risk_binned'] = pd.cut(
        full_df['environmental_risk_score'], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    
    cross_tab = pd.crosstab(
        full_df['segment_label'],
        full_df['env_risk_binned'],
        normalize='index'
    ) * 100

    print("\nBrand Cluster vs Environmental Risk Score (% distribution within each cluster):")
    print(cross_tab.round(1))

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    cross_tab.plot(kind='bar', stacked=True, ax=ax, colormap='RdYlGn_r')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Brand Clusters vs Company Environmental Risk Score', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Environmental Risk Score', bbox_to_anchor=(1.05, 1), fontsize=9)
    plt.xticks(rotation=0)  # Horizontal labels for short names
    plt.tight_layout()
    plt.show()
else:
    print("Required columns not found - run cluster labeling cell first")