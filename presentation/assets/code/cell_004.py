# Load the comprehensive brand information dataset
df = pd.read_csv('data/raw/brand_information.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns ({len(df.columns)} total):")
print(df.columns.tolist())

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Total brands: {len(df)}")
print(f"Unique companies: {df['company_name'].nunique()}")
print(f"Unique industries: {df['industry_name'].nunique()}")

print("\nFirst few rows:")
df.head()