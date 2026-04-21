"""
FHE Model Training & Compilation (Component 4)
Handles training, quantization, and TFHE circuit generation using Zama Concrete ML.
"""

from concrete.ml.sklearn import LogisticRegression
import numpy as np

class FHEModel:
    def __init__(self, n_bits=8):
        """
        n_bits: The number of bits for quantization. 
        8 bits balances predictive utility with FHE efficiency.
        """
        self.model = LogisticRegression(n_bits=n_bits)
        self.is_compiled = False

    def train(self, X_train, y_train):
        """Trains the model on cleartext data (standard training)."""
        print("[Component 4] Starting cleartext training...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def compile_to_fhe(self, X_train):
        """
        Compiles the trained model into an FHE circuit using TFHE.
        This enables prediction on fully encrypted medical data.
        """
        print("[Component 4] Compiling to FHE circuit via Zama (TFHE)...")
        self.model.compile(X_train)
        self.is_compiled = True
        print("FHE Compilation successful!")

    def predict(self, X_test, use_fhe=False):
        """
        Component 6: Secure Inference
        if use_fhe=True: Executes within the encrypted TFHE domain.
        if use_fhe=False: Executes as a quantized model in the clear.
        """
        if use_fhe and not self.is_compiled:
            raise RuntimeError("Model must be compiled before FHE execution.")
            
        fhe_mode = "execute" if use_fhe else "clear"
        return self.model.predict(X_test, fhe=fhe_mode)

    def get_accuracy(self, X_test, y_test):
        """Helper to verify model predictive utility."""
        y_pred = self.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        return accuracy
