# Extract feature matrix
X = full_df[feature_columns].copy()

# Handle any remaining missing values in feature matrix
X = X.fillna(X.median())

print(f"Feature shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Scale features using StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled shape: {X_scaled.shape}")