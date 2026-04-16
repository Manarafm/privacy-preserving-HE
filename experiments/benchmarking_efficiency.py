import time
import psutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Note: Run !pip install concrete-ml first in Colab if needed
!pip install -q concrete-ml
from concrete.ml.sklearn import LogisticRegression as ConcreteLR

print("Efficiency Metrics Evaluation on Heart Disease Dataset")
print("=" * 60)

# Load and preprocess Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

print("Loading Heart Disease dataset...")
df = pd.read_csv(url, names=column_names, na_values='?')

# Data preprocessing for Heart Disease
print(f"Dataset shape: {df.shape}")
print(f"Missing values before processing:\n{df.isnull().sum()}")

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

print(f"Missing values after processing:\n{df.isnull().sum()}")
print(f"Class distribution: {df['target'].value_counts().to_dict()}")

# Split features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")

# ========== Plaintext Inference Time ==========
print("\n" + "="*60)
print("Measuring Plaintext Inference Time...")

plain_model = LogisticRegression(max_iter=1000, random_state=42)
plain_model.fit(X_train, y_train)

# Warm-up predictions
for _ in range(5):
    plain_model.predict(X_test[0].reshape(1, -1))

# Measure plaintext inference time (50 samples)
plain_inf_times = []
for i, sample in enumerate(X_test[:50]):
    start = time.time()
    plain_model.predict(sample.reshape(1, -1))
    plain_inf_times.append(time.time() - start)

    # Progress indicator
    if (i+1) % 10 == 0:
        print(f"  Processed {i+1}/50 plaintext samples")

plain_avg_inf = np.mean(plain_inf_times)
plain_std_inf = np.std(plain_inf_times)
plain_total_time = sum(plain_inf_times)

print(f"Plaintext inference - Avg: {plain_avg_inf:.6f}s, Std: {plain_std_inf:.6f}s")
print(f"Total time for 50 samples: {plain_total_time:.3f}s")

# ========== FHE Setup and Inference Time ==========
print("\n" + "="*60)
print("Setting up FHE Model and Measuring Inference Time...")

# FHE model training and compilation
fhe_model = ConcreteLR(n_bits=8, max_iter=1000, random_state=42)
print("Training FHE model...")
fhe_model.fit(X_train, y_train)

print("Compiling FHE circuit...")
compilation_start = time.time()
fhe_model.compile(X_train[:50])  # Compilation
compilation_time = time.time() - compilation_start
print(f"FHE compilation time: {compilation_time:.2f}s")

# Warm-up FHE predictions
print("Warming up FHE execution...")
for _ in range(2):
    fhe_model.predict(X_test[0].reshape(1, -1), fhe="simulate")

# Measure FHE inference time (10 samples - real execute)
fhe_inf_times = []
print(f"\nExecuting real FHE inference on 10 samples (this will take a while)...")
for i in range(10):
    sample = X_test[i].reshape(1, -1)

    print(f"  Sample {i+1}/10 - Starting FHE execution...")
    start = time.time()
    fhe_model.predict(sample, fhe="execute")
    inf_time = time.time() - start
    fhe_inf_times.append(inf_time)

    print(f"    Completed in {inf_time:.2f}s")

fhe_avg_inf = np.mean(fhe_inf_times)
fhe_std_inf = np.std(fhe_inf_times)
fhe_total_time = sum(fhe_inf_times)

print(f"\nFHE inference - Avg: {fhe_avg_inf:.2f}s, Std: {fhe_std_inf:.2f}s")
print(f"Total FHE time for 10 samples: {fhe_total_time:.2f}s")

# ========== Resource Consumption ==========
print("\n" + "="*60)
print("Measuring Resource Consumption...")

process = psutil.Process()

# Baseline measurements
print("Taking baseline measurements...")
mem_before = process.memory_info().rss / (1024 ** 2)  # MB
cpu_before = psutil.cpu_percent(interval=0.5)

# Run one FHE prediction and measure resources
print("Running one FHE prediction with resource monitoring...")
start_time = time.time()
fhe_model.predict(X_test[0].reshape(1, -1), fhe="execute")
execution_time = time.time() - start_time

# Post-execution measurements
cpu_during = psutil.cpu_percent(interval=None)
mem_after = process.memory_info().rss / (1024 ** 2)
mem_delta_mb = mem_after - mem_before

print(f"Single FHE execution time: {execution_time:.2f}s")
print(f"CPU usage during execution: {cpu_during:.1f}%")
print(f"Memory delta: {mem_delta_mb:.2f} MB")

# ========== Communication Overhead (Federated Learning) ==========
print("\n" + "="*60)
print("Calculating Communication Overhead...")

# Model parameter size calculation
param_bytes = plain_model.coef_.nbytes + plain_model.intercept_.nbytes
print(f"Model parameters size: {param_bytes} bytes")

# For Federated Learning with 3 clients
n_clients = 3
comm_bytes_per_round = param_bytes * n_clients * 2  # Upload + download
comm_kb_total = comm_bytes_per_round / 1024
comm_mb_total = comm_kb_total / 1024

# Assuming 10 FL rounds
n_rounds = 10
total_comm_bytes = comm_bytes_per_round * n_rounds
total_comm_mb = total_comm_bytes / (1024 ** 2)

print(f"Communication per FL round (3 clients): {comm_kb_total:.2f} KB ({comm_mb_total:.2f} MB)")
print(f"Total communication for {n_rounds} FL rounds: {total_comm_mb:.2f} MB")

# ========== Power Consumption Estimate ==========
print("\n" + "="*60)
print("Estimating Power Consumption...")

# Rough power estimation
# Google Colab T4 GPU TDP: ~70W, CPU: ~15-25W
# Using CPU percentage for estimation
avg_cpu_usage = cpu_during
base_power_cpu = 20  # Watts for CPU
base_power_memory = 5  # Watts for memory
base_power_system = 10  # Watts for other components

# Power estimation formula
power_cpu = (avg_cpu_usage / 100) * base_power_cpu
power_memory = (mem_delta_mb / 100) * base_power_memory  # Rough scaling
power_total = power_cpu + power_memory + base_power_system

# Energy consumption for one inference
energy_joules = power_total * execution_time

print(f"Power consumption estimate:")
print(f"  CPU: {power_cpu:.2f} W")
print(f"  Memory: {power_memory:.2f} W")
print(f"  System: {base_power_system:.2f} W")
print(f"  Total: {power_total:.2f} W")
print(f"Energy per inference: {energy_joules:.3f} J")

# ========== Results Summary ==========
print("\n" + "="*60)
print("EFFICIENCY METRICS SUMMARY - HEART DISEASE DATASET")
print("="*60)

# Computational Overhead
if plain_avg_inf > 0:
    comp_overhead = fhe_avg_inf / plain_avg_inf
    comp_speedup = 1 / comp_overhead

    print(f"\n1. COMPUTATIONAL PERFORMANCE:")
    print(f"   Plaintext inference (avg): {plain_avg_inf*1000:.2f} ms")
    print(f"   FHE inference (avg): {fhe_avg_inf*1000:.2f} ms")
    print(f"   Computational Overhead: {comp_overhead:,.0f}×")
    print(f"   Speed Difference: {comp_speedup:.6f}×")
else:
    comp_overhead = "N/A"

print(f"\n2. RESOURCE CONSUMPTION:")
print(f"   CPU Usage: {cpu_during:.1f}%")
print(f"   Memory Increase: {mem_delta_mb:.2f} MB")
print(f"   FHE Compilation Time: {compilation_time:.2f}s")

print(f"\n3. COMMUNICATION OVERHEAD (Federated Learning):")
print(f"   Model Parameters: {param_bytes} bytes")
print(f"   Per FL Round (3 clients): {comm_kb_total:.2f} KB")
print(f"   Total for {n_rounds} rounds: {total_comm_mb:.2f} MB")

print(f"\n4. POWER & ENERGY:")
print(f"   Estimated Power: {power_total:.2f} W")
print(f"   Energy per FHE inference: {energy_joules:.3f} J")
print(f"   Energy for 10 inferences: {energy_joules*10:.3f} J")

print(f"\n5. DATASET SPECIFICS:")
print(f"   Dataset: Heart Disease (Cleveland)")
print(f"   Samples: {len(df)}")
print(f"   Features: {X.shape[1]}")
print(f"   Positive class (disease): {df['target'].sum()} samples")
print(f"   Negative class (no disease): {len(df) - df['target'].sum()} samples")

# ========== Comparative Analysis ==========
print("\n" + "="*60)
print("COMPARATIVE ANALYSIS")
print("="*60)

if plain_avg_inf > 0:
    # Time comparison table
    print(f"\nInference Time Comparison:")
    print(f"{'Metric':<25} {'Plaintext':<15} {'FHE':<15} {'Ratio':<10}")
    print(f"{'-'*65}")
    print(f"{'Avg time per sample':<25} {plain_avg_inf*1000:6.2f} ms    {fhe_avg_inf*1000:9.2f} ms    {comp_overhead:9,.0f}×")
    print(f"{'Time for 100 samples':<25} {plain_avg_inf*100*1000:6.0f} ms    {fhe_avg_inf*100*1000:9.0f} ms    {comp_overhead:9,.0f}×")

    # Practical implications
    print(f"\nPractical Implications:")
    print(f"• FHE is {comp_overhead:,.0f}× slower than plaintext inference")
    print(f"• 100 FHE inferences would take ~{fhe_avg_inf*100/60:.1f} minutes")
    print(f"• Equivalent plaintext inference takes ~{plain_avg_inf*100*1000:.0f} milliseconds")

    # Suitability assessment
    print(f"\nSuitability Assessment for Heart Disease Prediction:")
    print(f"✓ Suitable for: Batch processing, offline analysis, high-security scenarios")
    print(f"✗ Not suitable for: Real-time diagnosis, emergency care, mobile apps")
    print(f"✓ Good for: Clinical research, privacy-preserving studies, multi-institution collaboration")

print("\n" + "="*60)
print("Evaluation Complete!")
print("="*60)
