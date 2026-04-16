
import numpy as np
from sklearn.linear_model import LogisticRegression

def federated_train(X_train, y_train, n_clients=3):
    """
    Simulates Federated Learning (Horizontal) across multiple clients.
    Returns: Global model coefficients and intercept.
    """
    split_size = len(X_train) // n_clients
    weights = []
    intercepts = []

    for i in range(n_clients):
        start, end = i * split_size, (i+1) * split_size
        client_X, client_y = X_train[start:end], y_train[start:end]
        
        # Local training
        model = LogisticRegression(max_iter=1000).fit(client_X, client_y)
        weights.append(model.coef_)
        intercepts.append(model.intercept_)
    
    # Simple FedAvg (Averaging weights)
    global_coef = np.mean(weights, axis=0)
    global_inter = np.mean(intercepts, axis=0)
    
    return global_coef, global_inter
