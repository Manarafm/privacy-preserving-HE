import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import DataLoader

def run_mia_attack():
    print("=" * 70)
    print("PRIVACY ANALYSIS: MEMBERSHIP INFERENCE ATTACK (MIA)")
    print("=" * 70)

    # 1. Load Data
    loader = DataLoader(dataset="pima")
    X, _, y, _ = loader.load_and_split() # Get raw-ish scaled data

    # Split for MIA: Train (Target), Test (Non-members)
    X_target_train, X_attack_test, y_target_train, y_attack_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # 2. Train Victim Model (The "Plaintext" model)
    print("\n[Step 1] Training Victim Model...")
    victim = LogisticRegression(max_iter=1000)
    victim.fit(X_target_train, y_target_train)

    # 3. Simple Gap Attack (Measuring Overfitting)
    train_acc = accuracy_score(y_target_train, victim.predict(X_target_train))
    test_acc = accuracy_score(y_attack_test, victim.predict(X_attack_test))
    
    print(f"Victim Train Accuracy: {train_acc:.4f}")
    print(f"Victim Test Accuracy:  {test_acc:.4f}")
    print(f"Privacy Gap (Vulnerability): {train_acc - test_acc:.4f}")

    # 4. Confidence-Based Attack
    # Members (train set) usually have higher confidence scores
    print("\n[Step 2] Launching Membership Inference...")
    
    train_conf = np.max(victim.predict_proba(X_target_train), axis=1)
    test_conf = np.max(victim.predict_proba(X_attack_test), axis=1)

    # If confidence > threshold, we guess it was in the training set
    threshold = 0.75
    guessed_as_members = (train_conf > threshold).sum()
    guessed_non_members = (test_conf > threshold).sum()

    print(f"Attacker identified {guessed_as_members}/{len(train_conf)} members correctly.")
    print(f"Attacker falsely identified {guessed_non_members}/{len(test_conf)} non-members.")

    # 5. Conclusion
    print("\n" + "="*70)
    print("SECURITY ASSESSMENT")
    print("-" * 70)
    if (train_acc - test_acc) > 0.05:
        print("RESULT: Plaintext model is VULNERABLE to MIA.")
        print("MITIGATION: Deploying FHE-PHML is HIGHLY RECOMMENDED.")
    else:
        print("RESULT: Low Overfitting detected, but structural leakage remains possible.")
    print("="*70)

if __name__ == "__main__":
    run_mia_attack()
