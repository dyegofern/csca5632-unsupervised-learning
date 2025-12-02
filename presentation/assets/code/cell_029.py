n_components_to_interpret = min(10, X_pca.shape[1])
component_interpretations = {}
for i in range(n_components_to_interpret):
    pc_name = f'PC{i+1}'
    
    top_features_pc = top_features[
        (top_features['Component'] == pc_name) & 
        (top_features['Rank'] <= 5)
    ].sort_values('Abs_Loading', ascending=False)
    
    print(f"\n{pc_name} (explains {variance_df.iloc[i]['Variance_Explained']:.1f}% variance):")
    print("-" * 80)