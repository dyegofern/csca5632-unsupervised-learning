# Categorical distributions
print("="*60)
print("CATEGORICAL DISTRIBUTIONS")
print("="*60)

# Industry distribution
print("\nIndustry Distribution:")
print(df['industry_name'].value_counts().head(10))

# By Scope 1 + 2 Total
print("\nBy Scope 1 + 2 Total:")
print(df['scope12_total'].value_counts().sort_index())

# Greenwashing levels
print("\nInitial Greenwashing Level:")
print(df['initial_greenwashing_level'].value_counts().sort_index())

# Country of origin distribution
print("\nTop 10 Countries of Origin:")
print(df['country_of_origin'].value_counts().head(10))