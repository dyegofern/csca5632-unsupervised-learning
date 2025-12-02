    for _, row in top_features_pc.iterrows():
        direction = 'positively' if row['Loading'] > 0 else 'negatively'
        print(f"  {row['Feature']:50s} {row['Loading']:+.3f} ({direction})")

component_names = {
    'PC1': 'To be determined by loadings',  # Placeholder
    'PC2': 'To be determined by loadings',
    'PC3': 'To be determined by loadings',
    'PC4': 'To be determined by loadings',
    'PC5': 'To be determined by loadings',
}