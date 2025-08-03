import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv('jacket_sizes_data.csv')

# Features and target
X = df[['Age', 'Height_cm', 'Weight_kg']]
y = df['Jacket_Size_in']

# Step 1: Scale the features (very important for Kernel methods)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Create Kernel Ridge model with better parameters
# Try smaller alpha and gamma
model = KernelRidge(kernel='rbf', alpha=0.001, gamma=0.01)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Predict jacket size for a new person
# Remember to scale the input using the same scaler
new_person = [[25, 170, 65]]
new_person_scaled = scaler.transform(new_person)
predicted_size = model.predict(new_person_scaled)

print(f"Predicted Jacket Size: {predicted_size[0]:.2f} inches")
