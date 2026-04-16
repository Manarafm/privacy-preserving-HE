
### DESCRIPTION

This repository is the official implementation of the FHE-PHML framework, a 7-component architecture designed to secure medical machine learning workflows. While previous works relied on property-preserving or partially homomorphic encryption, FHE-PHML leverages Fully Homomorphic Encryption (FHE) via Zama Concrete ML.

This framework allows for decentralized Federated Learning and secure, encrypted inference on sensitive clinical data—specifically tested on the Pima Indians Diabetes and UCI Heart Disease datasets—without compromising predictive utility.

**The 7-Component Framework**

1: Local Training: Decentralized model training at healthcare sites.

2: FL Aggregation: Federated weight combination using PHE.

3: Global Model Consolidation: Creating the unified clinical model.

4: FHE Circuit Compilation: Converting the model into FHE circuits via Zama.

5: Multi-Dimensional Evaluation: Benchmarking Accuracy, Security, and Latency.

6: Secure Cloud Deployment : Private clinical predictions on patient data.

7: Encrypted Prediction and Local Decryption: Secure return of diagnosis to the end-user.

### DEVELOPMENT

This package requires **Python 3.12.x**. It is built on top of the **Zama Concrete ML** library, which utilizes the Rust compiler for FHE circuit compilation.

**Environment Setup**

Due to specific version requirements for FHE compatibility, we recommend a clean virtual environment:

```bash
# Install specific scikit-learn version first
pip install scikit-learn==1.5.0

# Install the framework and dependencies**
pip install -r requirements.txt
```

**Usage**

Run the Clinical Framework 
Execute the core pipeline (Federated training to FHE transition):

```Bash
python main.py
```

Run Research Benchmarks 
To reproduce the efficiency and privacy results presented in the paper.

```Bash

Benchmark 1: Efficiency (CPU, RAM, Energy, Latency)

python experiments/benchmarking_efficiency.py

Benchmark 2: Privacy (Membership Inference Attack Analysis)

python experiments/privacy_attack_mia.py

```

**Technical Foundation**

FHE-PHML utilizes the TFHE (Threshold Fully Homomorphic Encryption) scheme. Unlike legacy Order-Preserving Encryption (OPE) or Partially Homomorphic Encryption (PHE) used in isolation, our framework utilizes Zama’s Programmable Bootstrapping.

- Mathematical Privacy: Model parameters and patient inputs remain encrypted throughout the entire inference process.

- Non-Linearity: Allows for secure execution of non-linear activation functions (like the Logistic sigmoid) in the encrypted domain.

- Quantization: Uses 8-bit quantization to balance cryptographic security with predictive accuracy.

See [DEVELOPMENT](DEVELOPMENT.md) for detailed hardware specifications.

### SECURITY

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

### LICENSE

This project is licensed under the Apache-2.0 License.

### ATTRIBUTION & RESEARCH STATUS
This framework is the primary output of ongoing PhD research at **Cardiff University**. 

While this work is currently an **unpublished manuscript**, the authors kindly request that any use of this code, the 7-component architecture, or the benchmarking methodology be formally attributed to:
> **Alnasser, M., & Li, S. (2026). FHE-PHML Framework, Cardiff University.**

For inquiries regarding the full manuscript or collaboration, please contact the repository owner.
