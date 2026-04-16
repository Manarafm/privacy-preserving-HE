
"""
FHE Model Training
Handles training and FHE compilation using Concrete ML.
"""

from concrete.ml.sklearn import LogisticRegression
import numpy as np

class FHEModel:
    def __init__(self, n_bits=8):
        """
        n_bits: The number of bits for quantization. 
        8 bits is the standard for balancing accuracy and FHE speed.
        """
        self.model = LogisticRegression(n_bits=n_bits)
        self.is_compiled = False

    def train(self, X_train, y_train):
        """Trains the model on cleartext data (standard training)."""
        print("Starting cleartext training...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def compile_to_fhe(self, X_train):
        """
        Compiles the trained model into an FHE circuit.
        This allows the model to predict on encrypted data.
        """
        print("Compiling to FHE circuit (this may take a moment)...")
        self.model.compile(X_train)
        self.is_compiled = True
        print("Compilation complete!")

    def predict(self, X_test, use_fhe=False):
        """
        Predicts labels.
        if use_fhe=True: Executes as an encrypted circuit (Very slow, for testing).
        if use_fhe=False: Executes as a quantized model in the clear (Fast).
        """
        fhe_mode = "execute" if use_fhe else "clear"
        return self.model.predict(X_test, fhe=fhe_mode)

    def get_accuracy(self, X_test, y_test):
        """Helper to check model accuracy."""
        y_pred = self.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        return accuracy
