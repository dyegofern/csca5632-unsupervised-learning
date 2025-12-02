variance_df = reducer.get_variance_dataframe()
fig = viz.plot_variance_explained(variance_df, n_components=min(15, X_pca.shape[1]))
plt.show()