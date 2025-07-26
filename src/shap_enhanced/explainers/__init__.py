"""
SHAP Explainers Collection
==========================

Overview
--------

This subpackage contains a suite of SHAP-style explainers, each designed to handle different data structures,  
baseline strategies, and attribution mechanisms for interpretable machine learning. These explainers extend  
beyond standard SHAP to provide specialized techniques for:

- Temporal and sequential data (e.g., `TimeSHAP`, `LatentSHAP`)
- Sparse and discrete structures (e.g., `SPSHAP`, `SCSHAP`)
- Reinforcement learning–based strategies (e.g., `RLSHAP`)
- Surrogate model approximations (e.g., `SurroSHAP`)
- Multi-baseline and enhanced SHAP variants

Each module implements the `BaseExplainer` interface to ensure interoperability and consistent SHAP output.

Modules
-------

- **LatentSHAP**: Attribution in the latent space of an autoencoder.
- **TimeSHAP**: Pruned SHAP for long temporal sequences.
- **SurroSHAP**: Fast SHAP via surrogate regression.
- **MBSHAP**: Multi-baseline SHAP with per-sample reference sets.
- **RLSHAP**: SHAP value estimation via policy gradient masking.
- **SPSHAP**: Support-preserving masking for sparse inputs.
- **SCSHAP**: Sparse coalition enumeration for binary or one-hot inputs.
- **... and more**: Variants prefixed with a unique identifier (e.g., `ESSHAP`, `ERSHAP`) for experimentation.

Usage
-----

Import individual explainers or use the package to programmatically register all variants:

.. code-block:: python

    from shap_enhanced.explainers import LatentSHAP, TimeSHAP, SurroSHAP
"""

from .ABSHAP import AdaptiveBaselineSHAPExplainer
from .AttnSHAP import AttnSHAPExplainer
from .BSHAP import BShapExplainer
from .CASHAP import CoalitionAwareSHAPExplainer
from .CMSHAP import ContextualMaskingSHAPExplainer
from .ECSHAP import EmpiricalConditionalSHAPExplainer
from .ERSHAP import ERSHAPExplainer
from .ESSHAP import EnsembleSHAPWithNoise
from .hSHAP import HShapExplainer
from .LatentSHAP import LatentSHAPExplainer
from .MBSHAP import NearestNeighborMultiBaselineSHAP
from .RLSHAP import RLShapExplainer
from .SCSHAP import SparseCoalitionSHAPExplainer
from .SPSHAP import SupportPreservingSHAPExplainer
from .SurroSHAP import SurrogateSHAPExplainer
from .TimeSHAP import TimeSHAPExplainer

__all__ = [
    "AdaptiveBaselineSHAPExplainer",
    "AttnSHAPExplainer", 
    "BShapExplainer",
    "CoalitionAwareSHAPExplainer",
    "ContextualMaskingSHAPExplainer",
    "EmpiricalConditionalSHAPExplainer",
    "ERSHAPExplainer",
    "EnsembleSHAPWithNoise",
    "HShapExplainer",
    "LatentSHAPExplainer",
    "NearestNeighborMultiBaselineSHAP",
    "RLShapExplainer",
    "SparseCoalitionSHAPExplainer",
    "SupportPreservingSHAPExplainer",
    "SurrogateSHAPExplainer",
    "TimeSHAPExplainer",
]