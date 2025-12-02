# Check data types and missing values
print("Data types and missing values:")
info_df = pd.DataFrame({
    'Column': df.columns,
    'Non-Null Count': df.count(),
    'Null Count': df.isnull().sum(),
    'Dtype': df.dtypes
})
print(info_df.to_string())

# Summary statistics for key numeric columns
print("\n" + "="*60)
print("KEY NUMERIC VARIABLES")
print("="*60)

key_numeric = ['scope12_total', 'revenues', 'employees', 'revenue_billion_usd', 
               'electric_vehicles_percent', 'customer_loyalty_index']
available_numeric = [col for col in key_numeric if col in df.columns]

if available_numeric:
    print(df[available_numeric].describe())