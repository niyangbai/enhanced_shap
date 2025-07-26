# Contributing to Enhanced SHAP

Thank you for your interest in contributing to Enhanced SHAP! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please treat all contributors with respect and create a welcoming environment for everyone.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your feature or bug fix
5. Make your changes
6. Add tests and documentation
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/enhanced_shap.git
   cd enhanced_shap
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Contributing Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting  
- **MyPy**: Type checking

Run these tools before committing:
```bash
black src tests
ruff check src tests --fix
mypy src
```

### Commit Messages

Use clear and descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Branch Naming

Use descriptive branch names:
- `feature/add-new-explainer`
- `bugfix/fix-memory-leak`
- `docs/update-api-reference`

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_explainers.py

# Run tests matching a pattern
pytest -k "test_latent"
```

### Writing Tests

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Follow the existing test structure
- Test edge cases and error conditions

### Test Organization

```
tests/
├── test_base_explainer.py      # Base explainer tests
├── test_explainers.py          # Individual explainer tests
├── test_tools.py               # Tools module tests
├── test_edge_cases.py          # Edge case testing
└── conftest.py                 # Shared test fixtures
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: Description of when this is raised.
        
    Example:
        >>> example_function("hello", 42)
        True
    """
```

### Building Documentation

```bash
cd docs
make html
```

## Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass
   - Run code quality tools
   - Update documentation
   - Add entries to CHANGELOG.md

2. **Pull Request Requirements:**
   - Clear description of changes
   - Link to related issues
   - Include test coverage for new features
   - Update documentation as needed

3. **Review Process:**
   - Maintainers will review your PR
   - Address any feedback promptly
   - Be prepared to make changes

## Types of Contributions

### New Explainer Methods

When adding new explainer methods:

1. Inherit from `BaseExplainer`
2. Implement the `shap_values` method
3. Add comprehensive docstrings
4. Include usage examples
5. Add thorough tests
6. Update the explainers `__init__.py`

### Bug Fixes

- Include a test that reproduces the bug
- Ensure the fix doesn't break existing functionality
- Update documentation if the bug was in documented behavior

### Documentation Improvements

- Fix typos and grammatical errors
- Improve clarity and examples
- Add missing documentation
- Update outdated information

## Getting Help

- Create an issue for bugs or feature requests
- Join discussions in existing issues
- Ask questions in pull request reviews

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- GitHub contributors list
- Release notes for major contributions

Thank you for contributing to Enhanced SHAP!