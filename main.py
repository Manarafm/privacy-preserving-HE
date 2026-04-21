import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

# Install (run once)
!pip install -q concrete-ml torch scikit-learn pandas numpy matplotlib flwr

import concrete.ml
from concrete.ml.sklearn import LogisticRegression as ConcreteLR

print("Starting framework: FL + Real TFHE on Heart Disease Dataset")

# Load Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

df = pd.read_csv(url, names=column_names, na_values='?')

# Data preprocessing for Heart Disease
print(f"Dataset loaded: {df.shape}")
print(f"Missing values before processing:\n{df.isnull().sum()}")

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with median for numerical columns
for col in df.columns:
    if col != 'num':  # Don't fill target column
        df[col] = df[col].fillna(df[col].median())

print(f"\nMissing values after processing:\n{df.isnull().sum()}")

# Convert to binary classification (0 = no disease, 1-4 = disease)
df['target'] = (df['num'] > 0).astype(int)
df = df.drop('num', axis=1)

# Split features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset: {len(df)} samples")
print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
print(f"Training samples: {len(X_train_full)}")
print(f"Testing samples: {len(X_test)}")

# Plaintext baseline
plain_model = LogisticRegression(max_iter=1000, random_state=42)
plain_model.fit(X_train_full, y_train_full)
y_pred_plain = plain_model.predict(X_test)
plain_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_plain),
    "Precision": precision_score(y_test, y_pred_plain),
    "Recall": recall_score(y_test, y_pred_plain),
    "F1_Score": f1_score(y_test, y_pred_plain),
}
print("\n=== Plaintext Baseline ===")
print("Plaintext Metrics:", plain_metrics)

# Federated Learning (3 clients, FedAvg)
print("\n=== Federated Learning (3 clients) ===")
n_clients = 3
split_size = len(X_train_full) // n_clients
client_models = []

for i in range(n_clients):
    start = i * split_size
    end = (i+1) * split_size if i != n_clients-1 else len(X_train_full)
    client_X = X_train_full[start:end]
    client_y = y_train_full[start:end]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(client_X, client_y)
    client_models.append(model)
    print(f"Client {i+1} trained on {len(client_X)} samples")

# Federated averaging
avg_coef = np.mean([m.coef_ for m in client_models], axis=0)
avg_intercept = np.mean([m.intercept_ for m in client_models], axis=0)

fl_model = LogisticRegression(max_iter=1000, random_state=42)
fl_model.coef_ = avg_coef
fl_model.intercept_ = avg_intercept
fl_model.classes_ = np.array([0, 1])

y_pred_fl = fl_model.predict(X_test)
fl_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_fl),
    "Precision": precision_score(y_test, y_pred_fl),
    "Recall": recall_score(y_test, y_pred_fl),
    "F1_Score": f1_score(y_test, y_pred_fl),
}
print("FL Metrics:", fl_metrics)

# Real TFHE with Concrete ML (using FL weights)
print("\n=== FHE Encryption & Execution ===")
fhe_model = ConcreteLR(n_bits=8, max_iter=1000, random_state=42)
fhe_model.fit(X_train_full, y_train_full)  # Train on full data for calibration
fhe_model.coef_ = fl_model.coef_  # Integrate FL global model weights
fhe_model.intercept_ = fl_model.intercept_

# Compile FHE circuit
print("Compiling FHE circuit...")
fhe_circuit = fhe_model.compile(X_train_full[:50])  # Use subset for compilation
print("FHE circuit compiled successfully!")

# Real FHE execution on test samples
n_samples = 20
real_results = []
real_times = []

print(f"\nExecuting FHE inference on {n_samples} samples...")
for i in range(n_samples):
    sample = X_test[i:i+1]
    true_label = y_test[i]

    start = time.time()
    pred = fhe_model.predict(sample, fhe="execute")
    time_taken = time.time() - start

    real_times.append(time_taken)
    real_results.append((pred[0], true_label))

    if (i+1) % 5 == 0:  # Print progress every 5 samples
        print(f"  Processed {i+1}/{n_samples} samples")

# Calculate FHE metrics
if real_results:
    preds = [r[0] for r in real_results]
    labels = [r[1] for r in real_results]

    fhe_metrics = {
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds),
        "Recall": recall_score(labels, preds),
        "F1_Score": f1_score(labels, preds),
        "Avg_Time": np.mean(real_times),
        "Std_Time": np.std(real_times),
    }

    print("\n=== FHE Results ===")
    print("FHE Metrics:", fhe_metrics)
    print(f"Average inference time: {fhe_metrics['Avg_Time']:.3f}s ± {fhe_metrics['Std_Time']:.3f}s")
    print(f"Total FHE time for {n_samples} samples: {sum(real_times):.2f}s")

    # Agreement with plaintext
    plain_preds = plain_model.predict(X_test[:n_samples])
    agreement = np.mean(np.array(preds) == plain_preds)
    print(f"\nPlaintext-FHE Agreement: {agreement:.2%}")

    # Statistical comparison
    fhe_correct = [1 if p == l else 0 for p, l in zip(preds, labels)]
    plain_correct = [1 if p == l else 0 for p, l in zip(plain_preds, labels)]

    if len(set(fhe_correct)) > 1 and len(set(plain_correct)) > 1:
        t_stat, p_value = ttest_rel(fhe_correct, plain_correct)
        print(f"t-test: t={t_stat:.2f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("  Significant difference detected (p < 0.05)")
        else:
            print("  No significant difference detected")
    else:
        print("t-test: Cannot compute - all predictions are identical")

    # Visualization
    print("\n=== Visualization ===")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Metrics comparison
    metrics = ['Accuracy', 'F1_Score']
    x = np.arange(len(metrics))

    axes[0].bar(x - 0.3, [plain_metrics[m] for m in metrics], 0.3,
                label='Plaintext', color='blue', alpha=0.7)
    axes[0].bar(x, [fl_metrics[m] for m in metrics], 0.3,
                label='FL', color='green', alpha=0.7)
    axes[0].bar(x + 0.3, [fhe_metrics[m] for m in metrics], 0.3,
                label='FHE', color='red', alpha=0.7)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Inference time plot
    sample_numbers = list(range(1, n_samples + 1))
    axes[1].plot(sample_numbers, real_times, marker='o', linestyle='-',
                 color='purple', label='FHE Inference Time')
    axes[1].axhline(y=fhe_metrics['Avg_Time'], color='r', linestyle='--',
                   label=f'Avg: {fhe_metrics["Avg_Time"]:.3f}s')
    axes[1].set_xlabel('Sample Number')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('FHE Inference Times')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Dataset: Heart Disease (Cleveland) - {len(df)} samples")
    print(f"Features: {X.shape[1]} clinical features")
    print(f"Binary classification: 0=No Disease, 1=Disease")
    print(f"Plaintext Accuracy: {plain_metrics['Accuracy']:.3f}")
    print(f"FL Accuracy: {fl_metrics['Accuracy']:.3f}")
    print(f"FHE Accuracy: {fhe_metrics['Accuracy']:.3f}")

else:
    print("FHE execution failed or no results generated")

print("\nFramework execution completed successfully!")
