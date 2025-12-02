# Create a working copy
full_df = df.copy()

# Check missing values summary
missing_summary = pd.DataFrame({
    'column': full_df.columns,
    'missing_count': full_df.isnull().sum(),
    'missing_pct': (full_df.isnull().sum() / len(full_df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values('missing_pct', ascending=False)

print("Missing values statistics:")
print(missing_summary.to_string(index=False))

# Handle missing values
numeric_cols = full_df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if full_df[col].isnull().sum() > 0:
        # For ratio/percentage columns, fill with 0
        if any(keyword in col.lower() for keyword in ['percent', 'ratio', 'index']):
            full_df[col].fillna(0, inplace=True)
        else:
            # For count/size columns, fill with median
            full_df[col].fillna(full_df[col].median(), inplace=True)
            
print(f"Remaining missing values: {full_df.isnull().sum().sum()}")