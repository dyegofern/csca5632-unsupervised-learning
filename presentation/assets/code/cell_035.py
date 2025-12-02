# Apply t-SNE on PCA-reduced data (recommended workflow)
# This reduces computational cost and noise
X_tsne, tsne_model = reducer.apply_tsne(X_pca, n_components=2)