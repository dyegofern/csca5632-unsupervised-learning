    full_df['online_sales'].astype(float) +
    full_df['product_portfolio_diversity']
)
full_df['market_positioning'] = (
    np.log1p(full_df['market_cap_billion_usd']) * 0.5 +
    full_df['customer_loyalty_index'] * 0.3 +
    full_df['branding_innovation_level'] * 0.2
)
env_risk_cols = ['deforestation_risk', 'chemical_pollution_risk', 'fossil_fuel_reliance_encoded']
full_df['env_risk_concentration'] = full_df[env_risk_cols].mean(axis=1)

interaction_cols = ['emissions_intensity', 'esg_investment_ratio', 'brand_age_log', 
                   'revenue_per_employee', 'demographic_breadth', 'income_diversity',
                   'lifestyle_complexity', 'operational_complexity', 'market_positioning',
                   'env_risk_concentration']
print(full_df[interaction_cols].describe().round(3))

print(full_df.describe())