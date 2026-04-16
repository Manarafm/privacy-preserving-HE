import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

print("=" * 70)
print("MEMBERSHIP INFERENCE ATTACK ANALYSIS - HEART DISEASE DATASET")
print("=" * 70)

# Load and preprocess Heart Disease dataset
print("\n1. LOADING HEART DISEASE DATASET...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

df = pd.read_csv(url, names=column_names, na_values='?')

print(f"   Dataset shape: {df.shape}")
print(f"   Features: {len(column_names) - 1}")

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with median for numerical columns
for col in df.columns:
    if col != 'num':  # Don't fill target column
        df[col] = df[col].fillna(df[col].median())

# Convert to binary classification (0 = no disease, 1-4 = disease)
df['target'] = (df['num'] > 0).astype(int)
df = df.drop('num', axis=1)

# Show class distribution
class_dist = df['target'].value_counts()
print(f"   Class distribution: No Disease={class_dist.get(0, 0)}, Disease={class_dist.get(1, 0)}")

# Prepare features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# ========== DATA SPLITTING FOR MIA ==========
print("\n2. SPLITTING DATA FOR MEMBERSHIP INFERENCE ATTACK...")

# Split: 50% train (target model), 50% for shadow/attack
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# Split remaining: 50% shadow model, 50% attack test
X_shadow, X_attack, y_shadow, y_attack = train_test_split(
    X_remaining, y_remaining, test_size=0.5, random_state=42, stratify=y_remaining
)

print(f"   Target model training samples: {len(X_train)}")
print(f"   Shadow model training samples: {len(X_shadow)}")
print(f"   Attack evaluation samples: {len(X_attack)}")

# ========== TARGET MODEL (VICTIM) ==========
print("\n3. TRAINING TARGET MODEL (VICTIM)...")
target_model = LogisticRegression(max_iter=1000, random_state=42)
target_model.fit(X_train, y_train)

# Target model performance
y_train_pred = target_model.predict(X_train)
y_attack_pred = target_model.predict(X_attack)

train_acc = accuracy_score(y_train, y_train_pred)
attack_acc = accuracy_score(y_attack, y_attack_pred)

print(f"   Target model training accuracy: {train_acc:.4f}")
print(f"   Target model attack set accuracy: {attack_acc:.4f}")
print(f"   Target model overfitting indicator: {train_acc - attack_acc:.4f}")

# ========== SHADOW MODEL (ATTACKER'S MODEL) ==========
print("\n4. TRAINING SHADOW MODEL (ATTACKER'S SURROGATE)...")
shadow_model = LogisticRegression(max_iter=1000, random_state=42)
shadow_model.fit(X_shadow, y_shadow)

# Shadow model performance
shadow_train_acc = accuracy_score(y_shadow, shadow_model.predict(X_shadow))
shadow_attack_acc = accuracy_score(y_attack, shadow_model.predict(X_attack))
print(f"   Shadow model training accuracy: {shadow_train_acc:.4f}")
print(f"   Shadow model attack accuracy: {shadow_attack_acc:.4f}")

# ========== GENERATE ATTACK TRAINING DATA ==========
print("\n5. GENERATING ATTACK TRAINING DATA FROM SHADOW MODEL...")

# Get confidence scores from shadow model
shadow_train_proba = shadow_model.predict_proba(X_shadow)
shadow_attack_proba = shadow_model.predict_proba(X_attack)

# Use maximum predicted probability as confidence
shadow_train_conf = np.max(shadow_train_proba, axis=1)
shadow_attack_conf = np.max(shadow_attack_proba, axis=1)

# Create attack dataset: 1 = member (in shadow train), 0 = non-member
shadow_labels = np.concatenate([np.ones(len(X_shadow)), np.zeros(len(X_attack))])
shadow_confidences = np.concatenate([shadow_train_conf, shadow_attack_conf])

print(f"   Member samples (shadow train): {len(X_shadow)}")
print(f"   Non-member samples (shadow test): {len(X_attack)}")
print(f"   Total attack training samples: {len(shadow_labels)}")

# ========== SIMPLE THRESHOLD-BASED ATTACK ==========
print("\n6. TRAINING THRESHOLD-BASED ATTACK MODEL...")

# Find optimal threshold (members tend to have higher confidence)
member_confidences = shadow_confidences[shadow_labels == 1]
non_member_confidences = shadow_confidences[shadow_labels == 0]

# Try different percentile thresholds
best_accuracy = 0
best_threshold = 0.5

for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    threshold = np.percentile(member_confidences, percentile)
    predictions = (shadow_confidences > threshold).astype(int)
    acc = accuracy_score(shadow_labels, predictions)

    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold
        best_percentile = percentile

print(f"   Optimal threshold: {best_threshold:.4f} (percentile {best_percentile} of member confidences)")
print(f"   Attack model accuracy on shadow data: {best_accuracy:.4f}")

# ========== LAUNCH ATTACK ON TARGET MODEL ==========
print("\n7. LAUNCHING ATTACK ON TARGET MODEL...")

# Get confidence scores from target model on attack set
target_attack_proba = target_model.predict_proba(X_attack)
target_attack_conf = np.max(target_attack_proba, axis=1)

# Apply threshold attack
attack_predictions = (target_attack_conf > best_threshold).astype(int)
true_labels = np.ones(len(X_attack))  # All attack samples are non-members for the target model

# Calculate attack metrics
mia_accuracy = accuracy_score(true_labels, attack_predictions)
mia_precision = precision_score(true_labels, attack_predictions, zero_division=0)
mia_recall = recall_score(true_labels, attack_predictions, zero_division=0)
mia_f1 = f1_score(true_labels, attack_predictions, zero_division=0)

print(f"   Attack Accuracy: {mia_accuracy:.4f}")
print(f"   Attack Precision: {mia_precision:.4f}")
print(f"   Attack Recall: {mia_recall:.4f}")
print(f"   Attack F1-Score: {mia_f1:.4f}")

# ========== ADVANCED ATTACK (LOGISTIC REGRESSION ATTACKER) ==========
print("\n8. ADVANCED ATTACK: TRAINING LOGISTIC REGRESSION ATTACKER...")

# Create more sophisticated attack features
attack_features = np.column_stack([
    shadow_confidences,  # Maximum confidence
    np.min(shadow_model.predict_proba(np.concatenate([X_shadow, X_attack])), axis=1),  # Minimum confidence
    np.std(shadow_model.predict_proba(np.concatenate([X_shadow, X_attack])), axis=1),  # Confidence std
    np.abs(shadow_model.predict_proba(np.concatenate([X_shadow, X_attack]))[:, 0] - 0.5)  # Distance from 0.5
])

# Train logistic regression attack model
attack_model = LogisticRegression(max_iter=1000, random_state=42)
attack_model.fit(attack_features, shadow_labels)

# Test on target model
target_features = np.column_stack([
    target_attack_conf,
    np.min(target_attack_proba, axis=1),
    np.std(target_attack_proba, axis=1),
    np.abs(target_attack_proba[:, 0] - 0.5)
])

advanced_predictions = attack_model.predict(target_features)
advanced_accuracy = accuracy_score(true_labels, advanced_predictions)

print(f"   Advanced Attack Accuracy: {advanced_accuracy:.4f}")

# ========== FINAL ASSESSMENT ==========
print("\n" + "="*70)
print("FINAL ASSESSMENT")
print("="*70)

risk_level = "HIGH" if mia_accuracy > 0.7 else "MEDIUM" if mia_accuracy > 0.6 else "LOW"
tfhe_benefit = "CRITICAL" if mia_accuracy > 0.65 else "SIGNIFICANT" if mia_accuracy > 0.55 else "MODERATE"

print(f"\n• Plaintext Model Vulnerability: {risk_level}")
print(f"• TFHE Privacy Benefit: {tfhe_benefit}")
print(f"• Recommendation for Heart Disease Data: {'USE TFHE' if mia_accuracy > 0.55 else 'Consider TFHE for maximum privacy'}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
