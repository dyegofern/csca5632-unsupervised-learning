axes[1].hist(full_df['esg_premium_divergence'], bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero divergence')
axes[1].set_xlabel('ESG-Premium Divergence Score', fontsize=11)
axes[1].set_ylabel('Number of Brands', fontsize=11)
axes[1].set_title('ESG-Premium Alignment Distribution\n(Negative = Premium Brand, Poor ESG)',
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()