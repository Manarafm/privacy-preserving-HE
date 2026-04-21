import time
import psutil
import numpy as np
import os
import sys

# Add the root directory to path so we can use the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader
from src.models.fhe_model import FHEModel
from sklearn.linear_model import LogisticRegression

def run_efficiency_benchmark():
    print("=" * 60)
    print("FHE-PHML EFFICIENCY & RESOURCE BENCHMARKING")
    print("=" * 60)

    # 1. Load Data using your Framework's Loader
    loader = DataLoader(dataset="uci")
    X_train, X_test, y_train, y_test = loader.load_and_split()
    
    # 2. Plaintext Baseline
    print("\n[1/4] Measuring Plaintext Baseline...")
    plain_model = LogisticRegression(max_iter=1000)
    plain_model.fit(X_train, y_train)
    
    start_p = time.time()
    for sample in X_test[:50]:
        plain_model.predict(sample.reshape(1, -1))
    plain_latency = (time.time() - start_p) / 50
    print(f"Plaintext Avg Latency: {plain_latency*1000:.4f} ms")

    # 3. FHE Performance
    print("\n[2/4] Measuring FHE Performance (Zama Concrete ML)...")
    fhe_framework = FHEModel(n_bits=8)
    fhe_framework.train(X_train, y_train)
    fhe_framework.compile_to_fhe(X_train)

    fhe_times = []
    # We test 10 samples for real FHE execution (it is slow!)
    for i in range(10):
        start_f = time.time()
        fhe_framework.predict(X_test[i].reshape(1, -1), use_fhe=True)
        fhe_times.append(time.time() - start_f)
        print(f"  Sample {i+1}/10 encrypted inference: {fhe_times[-1]:.2f}s")

    avg_fhe_latency = np.mean(fhe_times)

    # 4. Resource Consumption
    print("\n[3/4] Measuring Hardware Resources...")
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    # Run one FHE inference to catch peak CPU/RAM
    fhe_framework.predict(X_test[0].reshape(1, -1), use_fhe=True)
    
    mem_after = process.memory_info().rss / (1024 ** 2)
    cpu_usage = psutil.cpu_percent(interval=1.0)

    # 5. Final Report
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("-" * 60)
    print(f"Computational Overhead: {avg_fhe_latency / plain_latency:,.0f}x slower")
    print(f"Memory Usage Increase:  {mem_after - mem_before:.2f} MB")
    print(f"CPU Utilization:        {cpu_usage}%")
    print(f"Security Domain:        128-bit TFHE")
    print("="*60)

if __name__ == "__main__":
    run_efficiency_benchmark()
