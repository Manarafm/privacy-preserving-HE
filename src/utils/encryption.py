
"""
Key Management & Encryption
Handles the client-side logic for FHE key generation and data encryption.
"""

import time

class FHEClientManager:
    def __init__(self, model):
        """
        model: The compiled FHE model from previous step.
        """
        self.model = model
        self.circuit = model.fhe_circuit
        
    def generate_keys(self):
        """Generates the private and public FHE keys."""
        print("Generating FHE keys...")
        start_time = time.time()
        self.circuit.keygen()
        duration = time.time() - start_time
        print(f"Keys generated in {duration:.2f} seconds.")
        return duration

    def encrypt_data(self, data):
        """Encrypts cleartext data into an FHE ciphertext."""
        return self.circuit.encrypt(data)

    def decrypt_prediction(self, ciphertext):
        """Decrypts the FHE result back into a cleartext prediction."""
        return self.circuit.decrypt(ciphertext)
