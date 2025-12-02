fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = axes[0].scatter(pca_scores_df['PC1'], pca_scores_df['PC2'], c=full_df['environmental_risk_score'], cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('PC1', fontweight='bold', fontsize=12)
axes[0].set_ylabel('PC2', fontweight='bold', fontsize=12)
axes[0].set_title('Brands in Latent Space (PC1 vs PC2)\nColored by Environmental Risk', fontweight='bold', fontsize=13)
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Environmental Risk Score')