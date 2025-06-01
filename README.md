<!-- SHAP-Enhanced: Advanced Explainability Toolkit -->

<div align="center">

# SHAP-Enhanced: Advanced Explainability Toolkit

<a href="https://www.gnu.org/licenses/agpl-3.0">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg?logo=open-source-initiative" alt="License: AGPL-3.0"/>
</a>
<a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python" alt="Python Version"/>
</a>
<a href="https://isocpp.org/">
    <img src="https://img.shields.io/badge/Code-C%2B%2B-orange.svg?logo=c%2B%2B" alt="C++ Code"/>
</a>
<a href="https://git-scm.com/">
    <img src="https://img.shields.io/badge/Git-Repository-orange.svg?logo=git" alt="Git"/>
</a>
<a href="https://pypi.org/project/shap-enhanced/">
    <img src="https://img.shields.io/pypi/v/shap-enhanced.svg?logo=python&label=PyPI" alt="PyPI version"/>
</a>
<a href="https://niyangbai.github.io/enhanced_shap/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-brightgreen?logo=github" alt="Docs: GitHub Pages"/>
</a>
<a href="https://www.sphinx-doc.org/">
    <img src="https://img.shields.io/badge/built%20with-Sphinx-89e?logo=sphinx" alt="Built with Sphinx"/>
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

You can install directly from PyPI (recommended if available):

```sh
pip install shap-enhanced
```

Or build the package locally:

```sh
git clone https://github.com/niyangbai/enhanced_shap.git
cd enhanced_shap
pip install -r requirements.txt
pip install .
```

## Contribution

We welcome contributions to <b>SHAP-Enhanced</b>! To contribute, please follow these steps:

<ul>
    <li><b>Fork</b> this repository and clone your fork.</li>
    <li><b>Create a branch</b> for your feature or bugfix.</li>
    <li>Add or update tests, and run <code>pytest</code> to ensure they pass.</li>
    <li><b>Commit</b> and <b>push</b> your changes, then open a <b>pull request</b> with a clear description.</li>
</ul>

<b>General Guidelines:</b>
<ul>
    <li>Follow the <b>PEP8 coding style</b>.</li>
    <li>Document all public methods, classes, and modules using <b>Sphinx-style docstrings</b>.</li>
    <li>Ensure your code is <b>properly typed</b> and includes type hints.</li>
    <li>Write clear and concise commit messages.</li>
    <li>Be respectful and collaborative in code reviews.</li>
</ul>


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