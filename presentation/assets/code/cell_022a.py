print("\n" + "="*60)
print("PREMIUM-ESG DIVERGENCE ANALYSIS")
print("="*60)
prem_div_cols = ['brand_name', 'premium_positioning', 'esg_premium_divergence']
print("\nHighest Premium-ESG Divergence (Premium brands from strong ESG companies):")
print(full_df.nlargest(10, 'esg_premium_divergence')[prem_div_cols].to_string(index=False))

print("\nLowest Premium-ESG Divergence (Premium brands from weak ESG companies):")
print(full_df.nsmallest(10, 'esg_premium_divergence')[prem_div_cols].to_string(index=False))