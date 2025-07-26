# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive `__all__` declarations across all modules
- Type hint support with `py.typed` marker file
- Professional package metadata and tooling configuration
- Development dependencies and optional dependency groups

### Changed
- Standardized package structure and imports
- Enhanced documentation strings across modules
- Version synchronization between `__init__.py` and `pyproject.toml`

### Fixed
- Import organization and consistency
- Package metadata completeness

## [0.0.1a4] - 2024-XX-XX

### Added
- Enhanced SHAP explainers collection with 16 different explainer variants
- Comprehensive tools module for evaluation, visualization, and benchmarking
- Base explainer interface for consistent API across all implementations
- Support for both sequential and tabular data explanations
- Synthetic data generators for testing and benchmarking
- Visualization utilities for publication-ready plots
- Ground-truth Shapley value estimation via Monte Carlo methods

### Explainers Included
- LatentSHAP: Attribution in latent space of autoencoders
- TimeSHAP: Pruned SHAP for long temporal sequences  
- SurroSHAP: Fast SHAP via surrogate regression
- MBSHAP: Multi-baseline SHAP with per-sample reference sets
- RLSHAP: SHAP value estimation via policy gradient masking
- SPSHAP: Support-preserving masking for sparse inputs
- SCSHAP: Sparse coalition enumeration for binary inputs
- And 9 additional experimental variants (ABSHAP, AttnSHAP, BSHAP, CASHAP, CMSHAP, ECSHAP, ERSHAP, ESSHAP, hSHAP)

### Tools Included
- Evaluation metrics (MSE, Pearson correlation) 
- Visualization suite (3D surfaces, bar plots, feature comparisons)
- Synthetic dataset generators (sequential and tabular)
- Predefined neural network models (LSTM, MLP)
- Timing utilities for performance profiling
- Comparison framework for benchmarking explainers

## [0.0.1a2] - Previous Release

### Added
- Initial implementation of core explainer framework
- Basic documentation structure

## [0.0.1a1] - Initial Release

### Added
- Initial project structure
- Basic explainer implementations
- Core functionality proof of concept