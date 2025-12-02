divergence_cols = ['brand_name', 'sustainability_positioning', 'company_env_risk', 'sustainability_divergence']
top_divergence = full_df.nlargest(10, 'sustainability_divergence')[divergence_cols]
print(top_divergence.to_string(index=False))

print("\n" + "="*60)
print("BRANDS WITH LOWEST SUSTAINABILITY DIVERGENCE")
print("(Parent company more environmentally conscious than brand positioning)")
print("="*60)
bottom_divergence = full_df.nsmallest(10, 'sustainability_divergence')[divergence_cols]
print(bottom_divergence.to_string(index=False))