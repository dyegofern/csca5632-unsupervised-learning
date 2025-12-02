scatter2 = axes[1].scatter(pca_scores_df['PC1'], pca_scores_df['PC3'], c=full_df['revenues'], cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5, norm=plt.cm.colors.LogNorm())
axes[1].set_xlabel('PC1', fontweight='bold', fontsize=12)
axes[1].set_ylabel('PC3', fontweight='bold', fontsize=12)
axes[1].set_title('Brands in Latent Space (PC1 vs PC3)\nColored by Revenue',
                  fontweight='bold', fontsize=13)
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Revenue (log scale)')

plt.tight_layout()
plt.show()