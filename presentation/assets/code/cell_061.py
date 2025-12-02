# Compare clustering algorithms

comparison_df = clusterer.compare_algorithms()
print("\nClustering Algorithm Comparison:")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)