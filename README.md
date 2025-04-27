# SHAP-Enhanced: Advanced Explainability Toolkit

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

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
setup.py                # Install Script (optional)
```


## 📚 Documentation Style

All functions, classes, and methods are documented using **Sphinx-style docstrings**, for example:

```python
def foo(a: int) -> str:
    """Convert an integer to a string.

    :param int a: An integer input.
    :return str: The string representation of the input.
    """
    return str(a)
```

- **Auto Documentation** is compatible with **Sphinx**.
- Every public method, class, and module is **properly typed** and documented.



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



## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).
See the LICENSE file for details.

> **Note**: Any modified versions must also be made publicly available under the same license if deployed.



## ✍️ About

Developed by **Niyang Bai**  
Master Thesis — **Enhanced SHAP for Sequential and Sparse Data in Predictive Process Monitoring**  
University: **Friedrich-Alexander-Universität Erlangen-Nürnberg**