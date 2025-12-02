# Export final results to CSV
output_cols = ['brand_name', 'company_name', 'industry_name',
               'kmeans_cluster', 'hierarchical_cluster', 'dbscan_cluster',
               'segment_label']

# Add optional columns if they exist
optional_cols = ['initial_greenwashing_level', 'scope12_total', 'revenues',
                'country_of_origin', 'electric_vehicles_percent',
                'major_sustainability_award_last5y']
output_cols.extend([col for col in optional_cols if col in full_df.columns])

export_df = full_df[output_cols]
export_df.to_csv('output/brand_clustering_results.csv', index=False)

print("Results exported to output/brand_clustering_results.csv")
print(f"\nExported {len(export_df)} brands with {len(output_cols)} columns")
print(f"\nColumns exported: {output_cols}")