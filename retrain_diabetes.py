import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the dataset
data = pd.read_csv('data/diabetes.csv')

# Features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(model, os.path.join(models_dir, 'diabetes_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'diabetes_scaler.pkl'))

print("Diabetes model and scaler retrained and saved successfully.")
print(f"Number of features: {X.shape[1]}")
