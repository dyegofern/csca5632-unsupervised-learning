fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(full_df['sustainability_divergence'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero divergence')
axes[0].set_xlabel('Sustainability Divergence Score', fontsize=11)
axes[0].set_ylabel('Number of Brands', fontsize=11)
axes[0].set_title('Sustainability Divergence Distribution\n(Positive = Potential Greenwashing)',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[0].axvspan(-1, -0.2, alpha=0.2, color='blue', label='Under-promoting')
axes[0].axvspan(0.2, 1, alpha=0.2, color='red', label='Over-claiming')