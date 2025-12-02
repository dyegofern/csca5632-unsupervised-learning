def encode_esg_programs(val):
def encode_sustainability_actions(val):
def encode_fossil_fuel(val):
def encode_diversity(val):
def encode_localization(val):

full_df['esg_programs_encoded'] = full_df['esg_programs'].apply(encode_esg_programs)
full_df['sustainability_actions_encoded'] = full_df['sustainability_actions'].apply(encode_sustainability_actions)
full_df['fossil_fuel_reliance_encoded'] = full_df['fossil_fuel_reliance'].apply(encode_fossil_fuel)
full_df['product_portfolio_diversity_encoded'] = full_df['product_portfolio_diversity'].apply(encode_diversity)
full_df['supply_chain_localization_encoded'] = full_df['supply_chain_localization'].apply(encode_localization)
