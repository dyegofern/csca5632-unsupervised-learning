# Plot dendrogram
if X_pca.shape[0] <= 50:
    fig = viz.plot_dendrogram(X_pca, method='ward')
    plt.show()
else:
    #plotting a random subset
    subset_indices = np.random.choice(X_pca.shape[0], size=50, replace=False)
    subset = X_pca[subset_indices]
    fig = viz.plot_dendrogram(subset, method='ward')
    plt.show()