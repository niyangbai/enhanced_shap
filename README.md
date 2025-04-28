# SHAP-Enhanced: Advanced Explainability Toolkit

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![C++ Code](https://img.shields.io/badge/Code-C%2B%2B-orange)](https://isocpp.org/)

> **Note**: Portions of this project, particularly the **core simulation engine** and **performance-critical algorithms**, were developed in **C++** for enhanced computational efficiency and integration with Python via **Pybind11**.

## Project Overview

**SHAP-Enhanced** is a **next-generation explainability framework** developed for **research-grade model interpretation**.
It is built to **extend** and **surpass** traditional methods like **SHAP**, **LIME**, and **Integrated Gradients**, offering **new algorithms**, **more complex simulation engines**, and **multimodal support** across:

- Tabular
- Sequential (Time-Series)
- Vision
- Sparse and Multimodal datasets

This framework is created following **PEP8 standards**, **Sphinx documentation format**, and is released under the **GNU Affero General Public License v3.0**.

Developed as part of a **Master Thesis**.

## ✨ Key Features

- ✅ Unified **Explainability Interface** (`BaseExplainer`)
- ✅ Extensive **Gradient-Based** and **Perturbation-Based** explainers
- ✅ **Sequential/Attention Explainability** (LSTM, Transformer)
- ✅ **Sparse Tabular** and **PCA Sparse** models
- ✅ **Tree-Based Explainers** (Advanced SHAP for Decision Trees)
- ✅ **Simulation Engine** with ground-truth SHAP validation
- ✅ **Evaluation Metrics**: MSE, Correlation, Advanced Metrics
- ✅ **Visualization Toolkit**: Feature Importance, Time Series, Attention, Interactions
- ✅ **Multimodal Models** (Tabular + Image + Text Fusion)
- ✅ 100% **Modular Design** for future methods
- ✅ **No external SHAP package dependency** — **pure internal kernels**



## 📦 Project Structure

```
src/shap_enhanced/
    explainers/         # All explanation modules
    algorithms/         # Core mathematical tools (kernels, integration, perturbation)
    models/             # Tabular, Sequential, Vision, Multimodal models
    simulation/         # Synthetic data, Evaluation, Explainer Comparison
    visualization/      # Plotting and Visualization utilities
    datasets/           # Synthetic Dataset Generators
    utils/              # Logger, Config Loader, Timer, Random Seed
examples/               # How-to Scripts
tests/                  # Organized Unit Tests
README.md               # This file
LICENSE                 # GNU Affero General Public License
requirements.txt        # Dependencies
```

## ⚙️ Installation

You can install the package locally:

```sh
git clone https://github.com/niyangbai/enhanced_shap.git
cd shap-enhanced
pip install -r requirements.txt
```

**Requirements** include:

- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pyyaml`

(See `requirements.txt` for full list.)



## 🚀 Quick Example
```python
from shap_enhanced.models.tabular.tabular_mlp import TabularMLP
from shap_enhanced.explainers.gradient_based.integrated_gradients_explainer import IntegratedGradientsExplainer
from shap_enhanced.datasets.synthetic_tabular import generate_tabular_data

# Generate synthetic data
X, y = generate_tabular_data(n_samples=100, n_features=10)

# Initialize model and explainer
model = TabularMLP(input_size=10, hidden_size=64, output_size=1)
explainer = IntegratedGradientsExplainer(model=model, baseline=torch.zeros(1, 10))

# Explain a sample
sample = X[0:1]
attributions = explainer.explain(sample)
print(attributions)
```


## 🧪 Simulation and Benchmarking

Evaluate your explainers against **true SHAP values**:

```python
from shap_enhanced.simulation.simulator import ModelSimulation

simulation = ModelSimulation(model=model, explainer=explainer, true_shap_function=lambda X, y: X)
mse = simulation.run_simulation()
print(f"Mean Squared Error: {mse:.4f}")
```

Use `simulation/comparison.py` to benchmark multiple explainers side-by-side!



## 🎨 Visualization

from shap_enhanced.visualization.feature_importance_plot import plot_feature_importance

plot_feature_importance(importances=attributions, feature_names=[f"Feature {i}" for i in range(10)])

Available visualization modules:
- `Feature Importance`
- `Time Series Attribution`
- `Attention Weights`
- `SHAP Interaction Effects`
- `Evaluation Metrics Comparison`

## 🤝 Contribution

We welcome contributions to **SHAP-Enhanced**! To contribute, please follow these steps:

1. **Fork the Repository**: Click the "Fork" button on the GitHub repository page.
2. **Clone Your Fork**: Clone your forked repository to your local machine:
    ```sh
    git clone https://github.com/niyangbai/enhanced_shap.git
    cd enhanced_shap
    ```
3. **Create a Branch**: Create a new branch for your feature or bug fix:
    ```sh
    git checkout -b feature-or-bugfix-name
    ```
4. **Make Changes**: Implement your changes, ensuring they follow **PEP8 standards** and include proper **Sphinx-style docstrings**.
5. **Write Tests**: Add or update unit tests in the `tests/` directory to cover your changes.
6. **Run Tests**: Ensure all tests pass before submitting your changes:
    ```sh
    pytest
    ```
7. **Commit Changes**: Commit your changes with a descriptive message:
    ```sh
    git add .
    git commit -m "Description of your changes"
    ```
8. **Push Changes**: Push your branch to your forked repository:
    ```sh
    git push origin feature-or-bugfix-name
    ```
9. **Submit a Pull Request**: Open a pull request to the main repository, describing your changes in detail.

### Contribution Guidelines
- Follow the **PEP8 coding style**.
- Document all public methods, classes, and modules using **Sphinx-style docstrings**.
- Ensure your code is **properly typed** and includes type hints.
- Write clear and concise commit messages.
- Be respectful and collaborative in code reviews.

Thank you for helping improve **SHAP-Enhanced**!

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).
See the LICENSE file for details.

> **Note**: Any modified versions must also be made publicly available under the same license if deployed.

## ✍️ About

Developed by **Niyang Bai**  
Master Thesis — **Enhanced SHAP for Sequential and Sparse Data in Predictive Process Monitoring**  
University: **Friedrich-Alexander-Universität Erlangen-Nürnberg**

## 📧 Contact

For inquiries, please reach out via GitHub Issues.
