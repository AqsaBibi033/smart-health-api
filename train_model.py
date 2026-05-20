import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data/health_data.csv")

# Features and target
X = df[['age', 'bmi', 'bp', 'glucose', 'smoking']]
y = df['risk']

# Encode target
y = y.map({'low': 0, 'medium': 1, 'high': 2})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model + scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model trained and saved successfully!")