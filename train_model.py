import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

print("Step 1: Loading data...")
try:
    df = pd.read_csv("creditcard.csv")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please download it from Kaggle and place it in the project folder.")
    exit()

print("Data loaded. Shape:", df.shape)

# --- Data Preprocessing ---
df = df.drop('Time', axis=1)
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop('Amount', axis=1)

print("Step 2: Preparing features and labels...")
FEATURES = [col for col in df.columns if col != 'Class']
X = df[FEATURES]
y = df['Class']

# Using a smaller sample of the data to make training and tuning faster
X_sample, _, y_sample, _ = train_test_split(X, y, stratify=y, random_state=42, test_size=0.8)
X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, stratify=y_sample, random_state=42, test_size=0.2)

# --- Hyperparameter Tuning ---
print("\nStep 3: Finding the best model using Hyperparameter Tuning...")
print("This may take several minutes...")

# Calculate scale_pos_weight for handling class imbalance
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight = neg_count / pos_count

# Define the base model
clf_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# Define the settings (hyperparameters) we want to test
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 0.8]
}

# Set up the search. We tell it to find the model with the best 'recall' score.
grid_search = GridSearchCV(
    estimator=clf_base,
    param_grid=param_grid,
    scoring='recall',  # Optimize for catching fraud
    cv=3,              # 3-fold cross-validation
    verbose=1,
    n_jobs=-1          # Use all available CPU cores
)

# Run the search on the training data
grid_search.fit(X_train, y_train)

# Get the best model found by the search
print("\nBest parameters found:", grid_search.best_params_)
clf = grid_search.best_estimator_

# --- Unsupervised Model ---
# This part remains the same
print("\nTraining the unsupervised Isolation Forest model...")
iso = IsolationForest(contamination='auto', random_state=42)
iso.fit(X_train)

# --- Saving Artifacts ---
print("\nStep 4: Saving artifacts...")
joblib.dump(clf, "model_xgb.pkl")
joblib.dump(iso, "model_iso.pkl")
joblib.dump(scaler, "scaler.pkl")

# We still set thresholds for the API, but our main evaluation is below
probs = clf.predict_proba(X_val)[:, 1]
t1 = float(np.percentile(probs[y_val==0], 98))
t2 = float(np.percentile(probs[y_val==1], 20))

with open("config.json", "w") as f:
    json.dump({"T1": t1, "T2": t2}, f)

print("✅ Models & artifacts saved successfully.")
print(f"Thresholds set: T1 (Approve) < {t1:.4f}, T2 (Hold) > {t2:.4f}")

# --- Model Evaluation ---
print("\n" + "="*50)
print("Step 5: Evaluating Tuned Model Performance on Validation Set")
print("="*50)

y_pred_val = clf.predict(X_val)
y_pred_proba_val = clf.predict_proba(X_val)[:, 1]

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred_val)
print(f"True Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}  <-- Frauds we missed!")
print(f"True Positives: {cm[1][1]}   <-- Frauds we caught!")

print("\nClassification Report:")
print(classification_report(y_val, y_pred_val, target_names=['Class 0 (Safe)', 'Class 1 (Fraud)']))

auc_roc = roc_auc_score(y_val, y_pred_proba_val)
print(f"\nAUC-ROC Score: {auc_roc:.4f}")

precision, recall, _ = precision_recall_curve(y_val, y_pred_proba_val)
auprc = auc(recall, precision)
print(f"AUPRC (Area Under Precision-Recall Curve): {auprc:.4f}")
print("\nNote: The new model was optimized to have a better 'Recall' for Class 1.")
print("="*50)