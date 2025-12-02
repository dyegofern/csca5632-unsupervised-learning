# Analyze top features per component
top_features = reducer.get_top_features_per_component(feature_columns, n_top=5)

print("Top 5 features for first 3 principal components:")
print(top_features[top_features['Component'].isin(['PC1', 'PC2', 'PC3'])])

# Plot feature importance for PC1
fig = viz.plot_feature_importance(top_features, component='PC1', n_features=10)
plt.show()