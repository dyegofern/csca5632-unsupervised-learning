# Analyze DBSCAN outliers (SECONDARY MODEL)
# These are brands that don't fit into any dense cluster

print(f"\n{'='*80}")
print("DBSCAN OUTLIER ANALYSIS")
print(f"{'='*80}")

outlier_data = full_df[full_df['dbscan_cluster'] == -1]
core_data = full_df[full_df['dbscan_cluster'] != -1]

print(f"\nOutliers: {len(outlier_data)} brands ({len(outlier_data)/len(full_df)*100:.1f}%)")
print(f"Core clusters: {len(core_data)} brands ({len(core_data)/len(full_df)*100:.1f}%)")

if len(outlier_data) > 0:
    print(f"\n{'='*80}")
    print("OUTLIER CHARACTERISTICS (vs. Core Brands)")
    print(f"{'='*80}")
    
    # Compare outliers to core brands
    if 'environmental_risk_score' in outlier_data.columns:
        print(f"\nEnvironmental Risk:")
        print(f"  Outliers mean: {outlier_data['environmental_risk_score'].mean():.3f}")
        print(f"  Core mean: {core_data['environmental_risk_score'].mean():.3f}")
    
    if 'revenues' in outlier_data.columns:
        print(f"\nRevenue:")
        print(f"  Outliers mean: ${outlier_data['revenues'].mean():,.0f}")
        print(f"  Core mean: ${core_data['revenues'].mean():,.0f}")
    
    if 'scope12_total' in outlier_data.columns:
        print(f"\nEmissions:")
        print(f"  Outliers mean: {outlier_data['scope12_total'].mean():,.0f}")
        print(f"  Core mean: {core_data['scope12_total'].mean():,.0f}")
    
    print(f"\nTop Industries in Outliers:")
    if 'industry_name' in outlier_data.columns:
        print(outlier_data['industry_name'].value_counts().head(5))
    
    print(f"\nSample Outlier Brands:")
    sample_cols = ['brand_name', 'company_name']
    if 'environmental_risk_score' in outlier_data.columns:
        sample_cols.append('environmental_risk_score')
    if 'revenues' in outlier_data.columns:
        sample_cols.append('revenues')
    available_sample_cols = [c for c in sample_cols if c in outlier_data.columns]
    print(outlier_data[available_sample_cols].head(10).to_string(index=False))

print(f"\n{'='*80}")
print("These outliers represent unusual brands that don't fit standard patterns.")
print("They may warrant individual investigation for unique characteristics.")
print(f"{'='*80}")
