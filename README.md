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


## Overview

**SHAP-Enhanced** is a research-focused Python library providing a unified, extensible platform for developing, benchmarking, and analyzing advanced SHAP (SHapley Additive exPlanations) variants for tabular, sequential, and sparse data.

- Implements state-of-the-art and experimental SHAP explainers with a clean, consistent interface.
- Supports **time series, LSTM, attention models, sparse (binary/one-hot) data, hierarchical and multi-baseline explainers**, and more.
- Built for scientific benchmarking, model debugging, and real-world explainability studies.

This framework is created following <b>PEP8 standards</b>, <b>Sphinx documentation format</b>, and is released under the <b>GNU Affero General Public License v3.0</b>.

## Installation

You can install the package locally:

```sh
git clone https://github.com/niyangbai/enhanced_shap.git
cd shap-enhanced
pip install -r requirements.txt
```

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