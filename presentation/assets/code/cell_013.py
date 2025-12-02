# Define feature groups
company_esg_features = ['scope12_total', ..., 'positive_innovation_risk']
demographic_features = ['income_middle', 'income_high', 'income_premium','income_snap_support']
lifestyle_features = []
operational_features = ['online_sales', ..., 'supply_chain_localization_encoded']
sustainability_features = ['electric_vehicles_percent', ..., 'sustainability_actions_encoded']
divergence_features = ['sustainability_positioning', ..., 'esg_premium_divergence']
market_features = ['customer_loyalty_index', ..., 'year_of_foundation']
interaction_features = ['emissions_intensity', ..., 'env_risk_concentration']

feature_columns = (company_esg_features + demographic_features + lifestyle_features + operational_features +  sustainability_features + divergence_features + market_features + interaction_features)