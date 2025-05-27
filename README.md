<!-- SHAP-Enhanced: Advanced Explainability Toolkit -->

<h1 align="center">
    SHAP-Enhanced: Advanced Explainability Toolkit
</h1>

<p align="center">
    <a href="https://www.gnu.org/licenses/agpl-3.0">
        <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg?logo=open-source-initiative" alt="License: AGPL-3.0"/>
    </a>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python" alt="Python Version"/>
    </a>
    <a href="https://isocpp.org/">
        <img src="https://img.shields.io/badge/Code-C%2B%2B-orange.svg?logo=c%2B%2B" alt="C++ Code"/>
    </a>
    <a href="https://git-scm.com/">
        <img src="https://img.shields.io/badge/Git-Repository-orange.svg?logo=git" alt="Git"/>
    </a>
</p>

<br>

<details>
    <summary><strong>Table of Contents</strong></summary>
    <ul>
        <li><a href="#project-overview">Project Overview</a></li>
        <li><a href="#key-features">Key Features</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#quick-example">Quick Example</a></li>
        <li><a href="#simulation-and-benchmarking">Simulation and Benchmarking</a></li>
        <li><a href="#visualization">Visualization</a></li>
        <li><a href="#contribution">Contribution</a></li>
        <li><a href="#license">License</a></li>
        <li><a href="#about">About</a></li>
        <li><a href="#contact">Contact</a></li>
    </ul>
</details>

---

## Project Overview

<blockquote>
    <b>SHAP-Enhanced</b> is a framework designed to support research in model interpretation. It builds upon methods like <b>SHAP</b>, <b>LIME</b>, and <b>Integrated Gradients</b>, introducing additional algorithms, simulation tools, and support for multimodal data across:
</blockquote>

<div align="center">

<table>
    <tr>
        <td align="center"><b>Tabular</b></td>
        <td align="center"><b>Sequential (Time-Series)</b></td>
        <td align="center"><b>Vision</b></td>
        <td align="center"><b>Sparse & Multimodal</b></td>
    </tr>
</table>

</div>

<p align="center">
    <img src="https://img.icons8.com/color/48/000000/brain.png" width="32" alt="Tabular"/>
    <img src="https://img.icons8.com/color/48/000000/line-chart.png" width="32" alt="Time-Series"/>
    <img src="https://img.icons8.com/color/48/000000/image.png" width="32" alt="Vision"/>
    <img src="https://img.icons8.com/color/48/000000/combo-chart.png" width="32" alt="Multimodal"/>
</p>

This framework is created following <b>PEP8 standards</b>, <b>Sphinx documentation format</b>, and is released under the <b>GNU Affero General Public License v3.0</b>.

<sub>Developed as part of a <b>Master Thesis</b>.</sub>

---

## Key Features

<ul>
    <li>Unified <b>Explainability Interface</b> (<code>BaseExplainer</code>)</li>
    <li>Extensive <b>Gradient-Based</b> and <b>Perturbation-Based</b> explainers</li>
    <li><b>Sequential/Attention Explainability</b> (LSTM, Transformer)</li>
    <li><b>Sparse Tabular</b> and <b>PCA Sparse</b> models</li>
    <li><b>Tree-Based Explainers</b> (Advanced SHAP for Decision Trees)</li>
    <li><b>Simulation Engine</b> with ground-truth SHAP validation</li>
    <li><b>Evaluation Metrics</b>: MSE, Correlation, Advanced Metrics</li>
    <li><b>Visualization Toolkit</b>: Feature Importance, Time Series, Attention, Interactions</li>
    <li><b>Multimodal Models</b> (Tabular + Image + Text Fusion)</li>
    <li>100% <b>Modular Design</b> for future methods</li>
    <li><b>No external SHAP package dependency</b> — <b>pure internal kernels</b></li>
</ul>

---

## Project Structure

```text
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

---

## Installation

You can install the package locally:

```sh
git clone https://github.com/niyangbai/enhanced_shap.git
cd shap-enhanced
pip install -r requirements.txt
```

**Requirements** include:

<div align="center">

<table>
    <tr>
        <td><code>torch</code></td>
        <td><code>numpy</code></td>
        <td><code>scikit-learn</code></td>
        <td><code>matplotlib</code></td>
        <td><code>seaborn</code></td>
        <td><code>pyyaml</code></td>
    </tr>
</table>

</div>

<sub>(See <code>requirements.txt</code> for full list.)</sub>

---

## Quick Example

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

---

## Simulation and Benchmarking

Evaluate your explainers against <b>true SHAP values</b>:

```python
from shap_enhanced.simulation.simulator import ModelSimulation

simulation = ModelSimulation(model=model, explainer=explainer, true_shap_function=lambda X, y: X)
mse = simulation.run_simulation()
print(f"Mean Squared Error: {mse:.4f}")
```

Use <code>simulation/comparison.py</code> to benchmark multiple explainers side-by-side!

---

## Visualization

```python
from shap_enhanced.visualization.feature_importance_plot import plot_feature_importance

plot_feature_importance(importances=attributions, feature_names=[f"Feature {i}" for i in range(10)])
```

Available visualization modules:
<ul>
    <li>Feature Importance</li>
    <li>Time Series Attribution</li>
    <li>Attention Weights</li>
    <li>SHAP Interaction Effects</li>
    <li>Evaluation Metrics Comparison</li>
</ul>

---

## Contribution

We welcome contributions to <b>SHAP-Enhanced</b>! To contribute, please follow these steps:

<ol>
    <li><b>Fork the Repository</b>: Click the "Fork" button on the GitHub repository page.</li>
    <li><b>Clone Your Fork</b>: Clone your forked repository to your local machine:
        <pre><code>git clone https://github.com/niyangbai/enhanced_shap.git
cd enhanced_shap
</code></pre>
    </li>
    <li><b>Create a Branch</b>: Create a new branch for your feature or bug fix:
        <pre><code>git checkout -b feature-or-bugfix-name
</code></pre>
    </li>
    <li><b>Make Changes</b>: Implement your changes, ensuring they follow <b>PEP8 standards</b> and include proper <b>Sphinx-style docstrings</b>.</li>
    <li><b>Write Tests</b>: Add or update unit tests in the <code>tests/</code> directory to cover your changes.</li>
    <li><b>Run Tests</b>: Ensure all tests pass before submitting your changes:
        <pre><code>pytest
</code></pre>
    </li>
    <li><b>Commit Changes</b>: Commit your changes with a descriptive message:
        <pre><code>git add .
git commit -m "Description of your changes"
</code></pre>
    </li>
    <li><b>Push Changes</b>: Push your branch to your forked repository:
        <pre><code>git push origin feature-or-bugfix-name
</code></pre>
    </li>
    <li><b>Submit a Pull Request</b>: Open a pull request to the main repository, describing your changes in detail.</li>
</ol>

<b>General Guidelines:</b>
<ul>
    <li>Follow the <b>PEP8 coding style</b>.</li>
    <li>Document all public methods, classes, and modules using <b>Sphinx-style docstrings</b>.</li>
    <li>Ensure your code is <b>properly typed</b> and includes type hints.</li>
    <li>Write clear and concise commit messages.</li>
    <li>Be respectful and collaborative in code reviews.</li>
</ul>

---

## License

This project is licensed under the <b>GNU Affero General Public License v3.0</b> (AGPL-3.0).  
See the LICENSE file for details.

<blockquote>
    <b>Note</b>: Any modified versions must also be made publicly available under the same license if deployed.
</blockquote>

---

## About

<ul>
    <li>Master Thesis — <b>Enhanced SHAP for Sequential and Sparse Data in Predictive Process Monitoring</b></li>
    <li>University: <b>Friedrich-Alexander-Universität Erlangen-Nürnberg</b></li>
</ul>

---

## Contact

For inquiries, please reach out via <a href="https://github.com/niyangbai/enhanced_shap/issues">GitHub Issues</a>.

