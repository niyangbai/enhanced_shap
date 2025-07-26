<!-- SHAP-Enhanced: Advanced Explainability Toolkit -->

<div align="center">

# SHAP-Enhanced: Advanced Explainability Toolkit

<a href="https://pypi.org/project/shap-enhanced/">
    <img src="https://img.shields.io/pypi/v/shap-enhanced.svg?logo=python&label=PyPI" alt="PyPI version"/>
</a>
<a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python" alt="Python Version"/>
</a>
<a href="https://github.com/niyangbai/enhanced_shap/actions/workflows/ci.yml">
    <img src="https://github.com/niyangbai/enhanced_shap/workflows/CI/badge.svg" alt="CI Status"/>
</a>
<a href="https://codecov.io/gh/niyangbai/enhanced_shap">
    <img src="https://codecov.io/gh/niyangbai/enhanced_shap/branch/main/graph/badge.svg" alt="Coverage"/>
</a>
<a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"/>
</a>
<a href="https://github.com/charliermarsh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" alt="Ruff"/>
</a>
<a href="https://mypy-lang.org/">
    <img src="https://img.shields.io/badge/mypy-checked-blue.svg" alt="MyPy checked"/>
</a>
<a href="https://www.gnu.org/licenses/agpl-3.0">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg?logo=open-source-initiative" alt="License: AGPL-3.0"/>
</a>
<a href="https://niyangbai.github.io/enhanced_shap/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-brightgreen?logo=github" alt="Docs: GitHub Pages"/>
</a>


</div>


## Overview

**SHAP-Enhanced** is a research-focused Python library providing a unified, extensible platform for developing, benchmarking, and analyzing advanced SHAP (SHapley Additive exPlanations) variants for tabular, sequential, and sparse data.

- Implements state-of-the-art and experimental SHAP explainers with a clean, consistent interface.
- Supports **time series, LSTM, attention models, sparse (binary/one-hot) data, hierarchical and multi-baseline explainers**, and more.
- Built for scientific benchmarking, model debugging, and real-world explainability studies.

This framework is created following <b>PEP8 standards</b>, <b>Sphinx documentation format</b>, and is released under the <b>GNU Affero General Public License v3.0</b>.

Official documentation is hosted on **GitHub Pages**:

- [https://niyangbai.github.io/enhanced_shap/](https://niyangbai.github.io/enhanced_shap/)

This site provides comprehensive API reference, tutorials, and examples.
Documentation is automatically generated from the source code and kept up to date with the latest release.


## Installation

### PyPI Installation (Recommended)

Install the latest stable version from PyPI:

```sh
pip install shap-enhanced
```

### Optional Dependencies

Install with additional features:

```sh
# Development dependencies (testing, linting, type checking)
pip install "shap-enhanced[dev]"

# Documentation dependencies
pip install "shap-enhanced[docs]"

# Example dependencies (Jupyter, visualization tools)
pip install "shap-enhanced[examples]"

# All dependencies
pip install "shap-enhanced[dev,docs,examples]"
```

### Development Installation

For contributing or local development:

```sh
git clone https://github.com/niyangbai/enhanced_shap.git
cd enhanced_shap
pip install -e ".[dev]"
```

## Contributing

We welcome contributions to **SHAP-Enhanced**! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### Quick Start for Contributors

1. **Fork** this repository and clone your fork
2. **Set up development environment**: `pip install -e ".[dev]"`
3. **Create a branch** for your feature or bugfix
4. **Make changes** and add tests
5. **Run quality checks**: `black src tests && ruff check src tests && mypy src`
6. **Run tests**: `pytest`
7. **Submit a pull request**

For detailed contribution guidelines, please see [CONTRIBUTING.md](https://github.com/niyangbai/enhanced_shap/blob/main/CONTRIBUTING.md).

### Code Quality Standards

- **Code Style**: We use [Black](https://github.com/psf/black) for formatting
- **Linting**: [Ruff](https://github.com/charliermarsh/ruff) for fast Python linting
- **Type Checking**: [MyPy](https://mypy-lang.org/) for static type analysis
- **Testing**: [Pytest](https://pytest.org/) with coverage reporting
- **Documentation**: Google-style docstrings with Sphinx


## License

This project is licensed under the <b>GNU Affero General Public License v3.0</b> (AGPL-3.0).  
See the LICENSE file for details.

<blockquote>
    <b>Note</b>: Any modified versions must also be made publicly available under the same license if deployed.
</blockquote>

## About

<ul>
    <li>Master Thesis — <b>Enhanced SHAP for Sequential and Sparse Data in Predictive Process Monitoring</b></li>
    <li>University: <b>Friedrich-Alexander-Universität Erlangen-Nürnberg</b></li>
</ul>

## Contact

For inquiries, please reach out via <a href="https://github.com/niyangbai/enhanced_shap/issues">GitHub Issues</a>.