"""
Key Management & Encryption
Handles the client-side logic for FHE key generation and data encryption.
"""
import time

class FHEClientManager:
    def __init__(self, model_wrapper):
        """
        model_wrapper: The FHEModel instance from models/fhe_model.py
        """
        # We access the Zama model inside your wrapper
        self.fhe_model = model_wrapper.model 
        
    def generate_keys(self):
        """Generates the private and public FHE keys on the client device."""
        print("[Security] Generating FHE keys (TFHE)...")
        start_time = time.time()
        
        # Accesses the underlying Concrete ML circuit keygen
        self.fhe_model.fhe_circuit.keygen()
        
        duration = time.time() - start_time
        print(f"Keys generated in {duration:.2f} seconds.")
        return duration

    def encrypt_data(self, data):
        """Encrypts cleartext data into an FHE ciphertext."""
        return self.fhe_model.fhe_circuit.encrypt(data)

    def decrypt_prediction(self, ciphertext):
        """Decrypts the FHE result back into a cleartext prediction."""
        return self.fhe_model.fhe_circuit.decrypt(ciphertext)
