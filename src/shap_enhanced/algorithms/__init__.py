"""
Enhanced SHAP Algorithms
========================

This module contains common algorithmic components extracted from SHAP explainers
to reduce code duplication and improve maintainability.

Core Components:
- coalition_sampling: Coalition generation strategies
- masking: Feature masking and imputation methods  
- shap_computation: Core SHAP value estimation algorithms
- normalization: Additivity constraint enforcement
- model_evaluation: Model output evaluation utilities
- data_processing: Input/output format handling
- attention: Attention/importance computation methods
"""

from .coalition_sampling import *
from .masking import *
from .shap_computation import *
from .normalization import *
from .model_evaluation import *
from .data_processing import *
from .attention import *