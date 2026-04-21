import numpy as np
from sklearn.linear_model import LogisticRegression

class FederatedAggregator:
    """
    Components 2 & 3: FL Aggregation & Global Model Consolidation
    Simulates decentralized training across healthcare sites and 
    combines weights using a simulated Federated Averaging (FedAvg) approach.
    """
    def __init__(self, n_clients=3):
        self.n_clients = n_clients

    def aggregate_local_weights(self, X_train, y_train):
        """
        Simulates Horizontal Federated Learning.
        Splits data into 'n' clients, trains local models, and averages parameters.
        """
        split_size = len(X_train) // self.n_clients
        weights = []
        intercepts = []

        for i in range(self.n_clients):
            # Define data shard for this specific client (site)
            start, end = i * split_size, (i+1) * split_size
            client_X, client_y = X_train[start:end], y_train[start:end]
            
            # Local training at the clinical site
            # We use a standard LogisticRegression to simulate the local step
            model = LogisticRegression(max_iter=1000).fit(client_X, client_y)
            
            weights.append(model.coef_)
            intercepts.append(model.intercept_)
        
        # Simple FedAvg: Global Model = Mean of all local model parameters
        global_coef = np.mean(weights, axis=0)
        global_inter = np.mean(intercepts, axis=0)
        
        print(f"Successfully aggregated parameters from {self.n_clients} simulated clients.")
        
        return global_coef, global_inter
