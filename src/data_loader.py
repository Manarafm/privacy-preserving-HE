import numpy as np
import os

def load_processed_data(dataset_name="pima"):
    """
    Loads pre-processed .npz files from the data folder.
    dataset_name: 'pima' or 'uci'
    """
    # Find the data directory relative to this script
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(base_path, f'processed_{dataset_name}.npz')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Ensure you uploaded the .npz files.")

    data = np.load(file_path)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

if __name__ == "__main__":
    # Test loading
    X_train, _, _, _ = load_processed_data("pima")
    print(f"Successfully loaded Pima data with shape: {X_train.shape}")
