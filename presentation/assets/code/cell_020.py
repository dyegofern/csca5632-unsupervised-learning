full_df['emissions_intensity'] = full_df['scope12_total'] / (full_df['revenues'] + 1)
full_df['esg_investment_ratio'] = full_df['r_and_d_spend_percent_revenue'] / (full_df['market_cap_billion_usd'] + 1)
current_year = 2025
full_df['brand_age'] = current_year - full_df['year_of_foundation']
full_df['brand_age_log'] = np.log1p(full_df['brand_age'])
full_df['revenue_per_employee'] = (full_df['revenues'] / (full_df['employees'] + 1)) / 1000
age_cols = ['age_prenatal', 'age_0_5', 'age_6_12', 'age_teens', 'age_young_adults', 'age_seniors']
full_df['demographic_breadth'] = full_df[age_cols].sum(axis=1)
income_cols = ['income_low', 'income_middle', 'income_high', 'income_premium']
full_df['income_diversity'] = full_df[income_cols].sum(axis=1)
lifestyle_cols = ['lifestyle_family', 'lifestyle_youth', 'lifestyle_seniors',
                 'lifestyle_health_focused', 'lifestyle_convenience',
                 'lifestyle_tech_savvy', 'lifestyle_sustainability_conscious']
full_df['lifestyle_complexity'] = full_df[lifestyle_cols].sum(axis=1)
full_df['operational_complexity'] = (
    full_df['has_franchises'].astype(float) + 
    full_df['owns_fleet'].astype(float) + 
    full_df['has_drive_through'].astype(float) +