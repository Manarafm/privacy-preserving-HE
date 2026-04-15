

## FHE-PHML: A Multi-Dimensional Framework for Privacy-Preserving Health Machine Learning Description

This repository is the official implementation of the FHE-PHML framework, a 7-component architecture designed to secure medical machine learning workflows. While previous works relied on property-preserving or partially homomorphic encryption, FHE-PHML leverages Fully Homomorphic Encryption (FHE) via Zama Concrete ML.

This framework allows for decentralized Federated Learning and secure, encrypted inference on sensitive clinical data—specifically tested on the Pima Indians Diabetes and UCI Heart Disease datasets—without compromising predictive utility.

**The 7-Component Framework**

Component 1: Local Training: Decentralized model training at healthcare sites.

Component 2: FL Aggregation: Federated weight combination using PHE.

Component 3: Global Model Consolidation: Creating the unified clinical model.

Component 4: FHE Circuit Compilation: Converting the model into FHE circuits via Zama.

Component 5: Multi-Dimensional Evaluation: Benchmarking Accuracy, Security, and Latency.

Component 6: Secure Cloud Deployment : Private clinical predictions on patient data.

Component 7: Encrypted Prediction and Local Decryption: Secure return of diagnosis to the end-user.

### Development

This package requires Python 3.12.12,. It is built on top of the Zama Concrete ML library, which requires the Rust compiler for FHE circuit compilation.

1. Install the core framework:

Bash
python -m pip install .
Note: This installs all core dependencies including concrete-ml, scikit-learn, and phe in the required order.

2. Run the Benchmarks (Component 5):
To verify the accuracy of the FHE circuits against plaintext models:

Bash
cd tests
python -m pytest
3. Run the Clinical Examples:
Install additional visualization tools and run the Pima or Heart Disease notebooks:

Bash
python -m pip install .[examples]
cd examples
#Execute a specific experiment via CLI or open in Jupyter
jupyter nbconvert --to notebook --execute Pima_Diabetes_FHE.ipynb --output Pima_Diabetes_FHE.ipynb
Technical Foundation
This framework utilizes the TFHE (Threshold Fully Homomorphic Encryption) scheme. Unlike legacy Order-Preserving Encryption (OPE), our use of Zama’s Programmable Bootstrapping ensures that the model structure and patient data remain mathematically private throughout the entire inference process.

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed hardware requirements.

### Security

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information on security disclosures and clinical data handling.

### License

This project is licensed under the Apache-2.0 License.
