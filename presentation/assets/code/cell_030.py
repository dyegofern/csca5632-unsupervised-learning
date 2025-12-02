n_top_features = 20
top_feature_names = (top_features.groupby('Feature')['Abs_Loading'].max().sort_values(ascending=False).head(n_top_features).index)

loading_matrix = pd.DataFrame(
    pca_model.components_[:n_components_to_interpret, :],
    columns=feature_columns,
    index=[f'PC{i+1}' for i in range(n_components_to_interpret)]
)[top_feature_names].T