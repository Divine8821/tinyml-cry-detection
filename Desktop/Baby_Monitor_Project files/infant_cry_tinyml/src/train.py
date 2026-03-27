import numpy as np
from dataset import load_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Load datasets
X_train, y_train = load_split("data/train")
X_val, y_val = load_split("data/validation")
X_test, y_test = load_split("data/test")

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# 🔍 VALIDATION (for tuning)
print("\nValidation Results:")
val_probs = model.predict_proba(X_val)[:,1]
threshold = 0.45  
val_pred = (val_probs > threshold).astype(int)
print(classification_report(y_val, val_pred))

# Save threshold
joblib.dump(threshold, "model/threshold.pkl")
print(f"Saved threshold: {threshold}")

# 🔍 TEST (final evaluation)
print("\nTest Results:")
test_pred = model.predict(X_test)
print(classification_report(y_test, test_pred))

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
