
from src.data.loader import get_dataloader
from src.federated.aggregator import federated_train
from src.models.fhe_model import FHEModel # Using the FHEModel class we created earlier
import time

def run_phd_framework(dataset="uci"):
    print(f"\n--- Framework Phase: {dataset.upper()} ---")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = get_dataloader(dataset)
    
    # 2. FL Component
    g_coef, g_inter = federated_train(X_train, y_train)
    
    # 3. FHE Component
    fhe_net = FHEModel(n_bits=8)
    fhe_net.train(X_train, y_train) # Calibration
    fhe_net.model.coef_ = g_coef   # Weight injection
    fhe_net.model.intercept_ = g_inter
    fhe_net.compile_to_fhe(X_train)
    
    # 4. Evaluation
    start = time.time()
    pred = fhe_net.predict(X_test[0:1], use_fhe=True)
    print(f"[{dataset}] Encrypted Result: {pred} | Time: {time.time()-start:.2f}s")

if __name__ == "__main__":
    for ds in ["uci", "pima"]:
        run_phd_framework(ds)
