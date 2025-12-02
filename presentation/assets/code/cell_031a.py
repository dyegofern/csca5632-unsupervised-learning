for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n{pc}:")
    print("\nHighest scoring brands:")
    print(pca_scores_df.nlargest(5, pc)[['brand_name', 'company_name', pc]])
    
    print("\nLowest scoring brands:")
    print(pca_scores_df.nsmallest(5, pc)[['brand_name', 'company_name', pc]])
    print("-" * 80)

full_df[[f'PC{i+1}' for i in range(n_components_to_interpret)]] = X_pca[:, :n_components_to_interpret]