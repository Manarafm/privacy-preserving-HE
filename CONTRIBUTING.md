# Contributing Guidelines

Thank you for your interest in this research project! This framework was developed as part of a PhD program at Cardiff University. We value contributions that push the boundaries of privacy-preserving machine learning in healthcare.

## 🔒 Security Model

Note for Researchers:
Unlike older property-preserving schemes, this framework primarily utilizes Fully Homomorphic Encryption (FHE) via Zama Concrete ML.

Security Standard: Our model aims for cryptographic semantic security where the model operator learns nothing about the underlying data values during inference.

## 🧪 Reporting Bugs & Feature Requests

We use GitHub Issues to track improvements. When filing an issue, please include:

- Environment Details: Confirm you are using Python 3.12.12 on macOS or Ubuntu.

- Dataset Context: Specify if the issue occurs with the Pima Indians or Heart Disease data components.

- Logs: Provide the error trace from the FHE compiler if applicable.

## 💻 Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

* You are working against the latest source on the main branch.

* You check existing open, or recently merged, pull requests to make sure someone else hasn't addressed the problem already.

* You open an issue to discuss any significant work—we would hate for your time to be wasted.

To send us a pull request, please:

* Fork the repository.

* Modify the source: Please focus on the specific change you are contributing. If you reformat all the code, it will be difficult to review your specific changes.

* Ensure local tests pass: Run the Component 5 benchmarks to verify accuracy.

* Commit to your fork using clear, descriptive commit messages.

* Send a pull request, answering any default questions in the pull request interface.

* Monitor CI: Pay attention to any automated failures reported in the pull request and stay involved in the conversation.

[!TIP]
GitHub provides additional documentation on forking a repository and creating a pull request.

## 🎓 Academic Integrity & Conduct

As a university-led project, we follow the standard Academic Code of Conduct. We expect all contributors to:

* Respect data privacy and clinical ethics.

* Provide honest and reproducible research results.

* Cite the original paper and Cardiff University if this work is used in other publications.

## ⚖️ Licensing
See the [LICENSE](LICENSE.txt) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
By contributing, you agree that your contributions will be licensed under the project's Apache-2.0 License.
