from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from pathlib import Path

cpp_modules = []

# --- Algorithms ---
algos = [
    "approximation", "attention", "distance_metrics", "integration",
    "interpolation", "masking", "perturbation", "sampling", "shapley_kernel"
]

for name in algos:
    src = Path("src/shap_enhanced/algorithms/_cpp") / f"{name}.cpp"
    if src.exists():
        cpp_modules.append(
            CppExtension(
                name=f"shap_enhanced.algorithms.{name}",
                sources=[str(src)],
            )
        )

# --- Datasets ---
datasets = [
    "synthetic_sequential", "synthetic_sparse", "synthetic_tabular"
]

for name in datasets:
    src = Path("src/shap_enhanced/datasets/_cpp") / f"{name}.cpp"
    if src.exists():
        cpp_modules.append(
            CppExtension(
                name=f"shap_enhanced.datasets.{name}",
                sources=[str(src)],
            )
        )

setup(
    name="shap_enhanced",
    version="0.1.0",
    packages=[
        "shap_enhanced",
        "shap_enhanced.algorithms",
        "shap_enhanced.datasets",
        "shap_enhanced.explainers",
        "shap_enhanced.models",
        "shap_enhanced.simulation",
        "shap_enhanced.utils",
        "shap_enhanced.visualization"
    ],
    package_dir={"": "src"},
    ext_modules=cpp_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
