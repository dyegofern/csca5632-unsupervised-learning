pca_scores_df = pd.DataFrame(
    X_pca[:, :n_components_to_interpret],
    columns=[f'PC{i+1}' for i in range(n_components_to_interpret)],
    index=full_df.index
)

pca_scores_df['brand_name'] = full_df['brand_name']
pca_scores_df['company_name'] = full_df['company_name']
pca_scores_df['industry_name'] = full_df['industry_name']