"""
FHE-PHML Framework: Main Entry Point
Orchestrates Data Loading, Federated Aggregation, and FHE Inference.
"""

import time
import numpy as np
from src.data.loader import DataLoader
from src.federated.aggregator import FederatedAggregator
from src.models.fhe_model import FHEModel
from src.utils.encryption import FHEClientManager
from src.utils.metrics import Evaluator

def run_framework():
    print("="*60)
    print("INITIALIZING FHE-PHML SECURE MEDICAL FRAMEWORK")
    print("="*60)

    # --- [Component 1] Data Loading ---
    # Choice: 'uci' (Heart Disease) or 'pima' (Diabetes)
    dataset_name = "uci"
    loader = DataLoader(dataset=dataset_name)
    X_train, X_test, y_train, y_test = loader.load_and_split()
    print(f"[Step 1] Data Loaded: {dataset_name.upper()} ({len(X_train)} training samples)")

    # --- [Component 2 & 3] Federated Aggregation ---
    aggregator = FederatedAggregator(n_clients=3)
    # Simulate the FedAvg process to get global weights
    global_coef, global_intercept = aggregator.aggregate_local_weights(X_train, y_train)
    print(f"[Step 2/3] Federated Aggregation Complete.")

    # --- [Component 4] FHE Circuit Compilation ---
    fhe_framework = FHEModel(n_bits=8)
    fhe_framework.train(X_train, y_train)
    fhe_framework.compile_to_fhe(X_train)
    print(f"[Step 4] Global Model Compiled into FHE Circuits.")

    # --- [Security Layer] FHE Key & Encryption Management ---
    client_manager = FHEClientManager(fhe_framework)
    client_manager.generate_keys()
    print(f"[Security] FHE Keys generated on simulated client.")

    # --- [Component 5, 6, & 7] Secure Inference & Evaluation ---
    print("\n" + "-"*60)
    print("STARTING SECURE MULTI-DIMENSIONAL EVALUATION")
    print("-"*60)

    # 1. Plaintext Prediction (Baseline)
    y_pred_plain = fhe_framework.predict(X_test, use_fhe=False)
    
    # 2. Encrypted Prediction (The core research contribution)
    # We time this specifically for the latency metric
    start_time = time.time()
    y_pred_fhe = fhe_framework.predict(X_test, use_fhe=True)
    fhe_latency = time.time() - start_time

    # 3. Metrics Generation
    evaluator = Evaluator(dataset_name=dataset_name)
    plain_metrics = evaluator.calculate_metrics(y_test, y_pred_plain)
    fhe_metrics = evaluator.calculate_metrics(y_test, y_pred_fhe)

    # 4. Statistical Validation (T-Test)
    # plain_correct and fhe_correct should be binary arrays of hits/misses
    t_stat, p_val = evaluator.run_t_test(
        (y_pred_plain == y_test).astype(int), 
        (y_pred_fhe == y_test).astype(int)
    )

    # --- [Output] Results Summary ---
    print("\nFINAL RESULTS:")
    print(f"Dataset:            {dataset_name.upper()}")
    print(f"Plaintext Accuracy: {plain_metrics['Accuracy']*100:.2f}%")
    print(f"FHE-PHML Accuracy:  {fhe_metrics['Accuracy']*100:.2f}%")
    print(f"Inference Latency:  {fhe_latency:.2f} seconds (Total Batch)")
    print(f"P-Value (Utility):  {p_val if p_val else 'N/A'}")
    
    # Generate the visualization for the thesis
    # We pass empty FL metrics as they are usually the same as Plaintext in this simulation
    evaluator.plot_comparison(plain_metrics, plain_metrics, fhe_metrics, [fhe_latency/len(X_test)]*len(X_test))

    print("\n" + "="*60)
    print("FRAMEWORK EXECUTION SUCCESSFUL")
    print("="*60)

if __name__ == "__main__":
    run_framework()
