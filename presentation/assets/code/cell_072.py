# Analyze the relationship between brand clusters and combined greenwashing risk
greenwash_col = 'environmental_risk_score' if 'environmental_risk_score' in full_df.columns else 'initial_greenwashing_level'

if greenwash_col in full_df.columns and 'segment_label' in full_df.columns:
    # Create risk categories for better visualization
    def categorize_risk(value):
        if value > 0.7:
            return 'High (>0.7)'
        elif value >= 0.3:
            return 'Medium (0.3-0.7)'
        else:
            return 'Low (<0.3)'
    
    full_df['risk_category'] = full_df[greenwash_col].apply(categorize_risk)
    
    # Cross-tabulation
    cross_tab = pd.crosstab(
        full_df['segment_label'],
        full_df['risk_category'],
        normalize='index'
    ) * 100
    
    # Ensure columns are in the right order
    col_order = ['Low (<0.3)', 'Medium (0.3-0.7)', 'High (>0.7)']
    cross_tab = cross_tab.reindex(columns=[c for c in col_order if c in cross_tab.columns])

    print(f"\nBrand Cluster vs {greenwash_col.replace('_', ' ').title()}")
    print("(% distribution within each cluster):")
    print(cross_tab.round(1))

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    cross_tab.plot(kind='bar', stacked=True, ax=ax, 
                   color=['lightgreen', 'gold', 'salmon'])
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title(f'Brand Clusters vs {greenwash_col.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Risk Level', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Also show average risk per cluster
    print(f"\nAverage {greenwash_col.replace('_', ' ').title()} per Cluster:")
    cluster_risk_avg = full_df.groupby('segment_label')[greenwash_col].agg(['mean', 'std', 'min', 'max'])
    print(cluster_risk_avg.round(3))
    
else:
    print("Required columns not found - run previous cells first")