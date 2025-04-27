from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line and not line.startswith("#")]

setup(
    name="shap_enhanced",
    version="0.0.1b",
    description="SHAP-Enhanced: Advanced Explainability Toolkit for Tabular, Sequential, Vision, and Multimodal Models",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/niyangbai/enhanced_shap",
    license="AGPL-3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)