# Analyze Hierarchical clusters (PRIMARY MODEL)

print(f"\n{'='*80}")
print("HIERARCHICAL CLUSTERING: SEGMENT ANALYSIS")
print(f"{'='*80}")

for cluster_id in sorted(full_df['hierarchical_cluster'].unique()):
    cluster_data = full_df[full_df['hierarchical_cluster'] == cluster_id]
    
    print(f"\n{'='*80}")
    print(f"SEGMENT {cluster_id} (n={len(cluster_data)} brands, {len(cluster_data)/len(full_df)*100:.1f}% of total)")
    print(f"{'='*80}")
    
    # Industry distribution
    if 'industry_name' in cluster_data.columns:
        print("\nTop Industries:")
        print(cluster_data['industry_name'].value_counts().head(5))
    
    # Environmental metrics
    if 'environmental_risk_score' in cluster_data.columns:
        print("\nEnvironmental Risk Distribution:")
        print(f"  Mean: {cluster_data['environmental_risk_score'].mean():.3f}")
        print(f"  Median: {cluster_data['environmental_risk_score'].median():.3f}")
        print(f"  Std: {cluster_data['environmental_risk_score'].std():.3f}")
    
    # Average emissions and revenue
    if 'scope12_total' in cluster_data.columns and 'revenues' in cluster_data.columns:
        avg_emissions = cluster_data['scope12_total'].mean()
        avg_revenue = cluster_data['revenues'].mean()
        print(f"\nOperational Scale:")
        print(f"  Avg Emissions (Scope 1+2): {avg_emissions:,.0f} metric tons")
        print(f"  Avg Revenue: ${avg_revenue:,.0f}")
    
    # Demographics summary
    demo_cols = [c for c in cluster_data.columns if c.startswith('target_age_') or c.startswith('target_income_')]
    if demo_cols:
        print("\nTarget Demographics (>30% prevalence):")
        for col in demo_cols:
            pct = cluster_data[col].sum()/len(cluster_data)*100
            if pct > 30:
                print(f"  {col}: {pct:.0f}%")
    
    # Sustainability features
    if 'electric_vehicles_percent' in cluster_data.columns:
        avg_ev = cluster_data['electric_vehicles_percent'].mean()
        print(f"\nSustainability Metrics:")
        print(f"  Avg Electric Vehicle Adoption: {avg_ev:.1f}%")
    
    if 'esg_program_present' in cluster_data.columns:
        esg_pct = cluster_data['esg_program_present'].sum()/len(cluster_data)*100
        print(f"  ESG Programs: {esg_pct:.0f}%")
    
    # Sample brands
    print("\nRepresentative Brands:")
    sample_cols = ['brand_name', 'company_name']
    if 'environmental_risk_score' in cluster_data.columns:
        sample_cols.append('environmental_risk_score')
    available_sample_cols = [c for c in sample_cols if c in cluster_data.columns]
    print(cluster_data[available_sample_cols].head(8).to_string(index=False))

print(f"\n{'='*80}")
print("END OF HIERARCHICAL SEGMENT ANALYSIS")
print(f"{'='*80}")
