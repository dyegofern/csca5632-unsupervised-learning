greenwashing_combined = (initial_greenwashing_risk + ... + greenwashing_factor_4).clip(lower=0, upper=1)

other_risks_combined = full_df['deforestation_risk'].fillna(0) + ... + (full_df['product_portfolio_diversity'].fillna(0) / 10)

env_risk_combined = greenwashing_combined + other_risks_combined
full_df['environmental_risk_score'] = env_risk_combined