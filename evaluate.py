import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

print("Step 1: Loading saved artifacts (model and scaler)...")
# 1. Load the trained model and the scaler
try:
    model = joblib.load("model_xgb.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: model_xgb.pkl or scaler.pkl not found.")
    print("Please run train_model.py first to create these files.")
    exit()

print("Step 2: Loading the dataset for evaluation...")
# 2. Load the dataset
try:
    df = pd.read_csv("creditcard.csv")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found.")
    exit()

print("Step 3: Applying the EXACT same preprocessing as in training...")
# 3. Apply the same preprocessing steps
# Drop 'Time' column if you did during training
df = df.drop('Time', axis=1)

# Use the LOADED scaler to transform the 'Amount' column
df['scaled_amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))

# Drop the original 'Amount' column
df = df.drop('Amount', axis=1)

print("Step 4: Separating features (X) and labels (y)...")
# 4. Prepare the feature set (X) and true labels (y)
# Ensure the columns in X are in the same order as during training
FEATURES = [col for col in df.columns if col not in ['Class']]
X_test = df[FEATURES]
y_test = df['Class']

print("Step 5: Making predictions on the test data...")
# 5. Make predictions
predictions = model.predict(X_test)

print("\n" + "="*50)
print("             Model Evaluation Report")
print("="*50)

# 6. Print the evaluation report
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(f"True Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}  <-- Frauds we missed!")
print(f"True Positives: {cm[1][1]}   <-- Frauds we caught!")

print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Class 0 (Safe)', 'Class 1 (Fraud)']))
print("="*50)