# Publishing Guide

This document explains how to publish new releases of the Enhanced SHAP package.

## Prerequisites

Before using the publishing script, ensure you have the following tools installed:

### Required Tools

1. **Python 3.10+** with pip
2. **Git** - Version control
3. **GitHub CLI (gh)** - For creating GitHub releases
   ```bash
   # Install on macOS
   brew install gh
   
   # Install on Ubuntu/Debian
   curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
   sudo apt update && sudo apt install gh
   ```

4. **Documentation Tools** (for docs publishing):
   ```bash
   pip install ghp-import
   pip install -e ".[docs]"  # Sphinx and related tools
   ```

5. **Package Publishing Tools**:
   ```bash
   pip install build twine
   ```

### Authentication Setup

1. **GitHub CLI Authentication**:
   ```bash
   gh auth login
   ```

2. **PyPI Authentication**:
   - Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
   - Generate API tokens for both
   - Configure twine:
     ```bash
     # ~/.pypirc
     [distutils]
     index-servers =
         pypi
         testpypi

     [pypi]
     username = __token__
     password = pypi-your-api-token-here

     [testpypi]
     repository = https://test.pypi.org/legacy/
     username = __token__
     password = pypi-your-testpypi-token-here
     ```

## Release Process

### 1. Prepare for Release

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.0.1a4"  # or whatever the next version should be
   ```

2. **Update CHANGELOG.md** with new features, fixes, and changes

3. **Commit all changes**:
   ```bash
   git add .
   git commit -m "Prepare release v0.0.1a4"
   git push
   ```

### 2. Publishing Options

The `publish.sh` script provides several publishing modes:

#### Option 1: Full Release (TestPyPI)
```bash
./publish.sh
```
This will:
- ✅ Build and deploy documentation to GitHub Pages
- ✅ Build Python package (wheel + sdist)
- ✅ Upload to **TestPyPI** (safe for testing)
- ✅ Create and push Git tag
- ✅ Create GitHub release with artifacts

#### Option 2: Production Release (PyPI)
```bash
./publish.sh --prod
```
⚠️ **Warning**: This publishes to production PyPI and cannot be undone!

#### Option 3: Documentation Only
```bash
./publish.sh --docs-only
```
Only builds and deploys documentation, useful for doc updates.

#### Option 4: Package Only
```bash
./publish.sh --package-only
```
Only builds and publishes package, skips documentation.

### 3. Verify Release

After publishing, verify:

1. **Documentation**: Check [GitHub Pages](https://niyangbai.github.io/enhanced_shap/)
2. **Package**: Test installation:
   ```bash
   # From TestPyPI
   pip install -i https://test.pypi.org/simple/ shap-enhanced
   
   # From PyPI (production)
   pip install shap-enhanced
   ```
3. **GitHub Release**: Check the [releases page](https://github.com/niyangbai/enhanced_shap/releases)

## Script Features

### Safety Features

- **Dependency checking** - Verifies required tools are installed
- **Git status check** - Warns about uncommitted changes
- **Tag validation** - Prevents duplicate releases
- **Production confirmation** - Requires explicit confirmation for PyPI uploads
- **Error handling** - Stops execution on any failure

### Colored Output

The script uses color-coded output for clarity:
- 🔵 **[INFO]** - General information
- 🟢 **[SUCCESS]** - Successful operations
- 🟡 **[WARNING]** - Important warnings
- 🔴 **[ERROR]** - Errors that stop execution

### Command Line Options

```bash
./publish.sh [OPTIONS]

Options:
  --prod         Upload to production PyPI (default: TestPyPI)
  --docs-only    Only build and deploy documentation
  --package-only Only build and publish package (skip docs)
  --help, -h     Show help message
```

## Troubleshooting

### Common Issues

1. **"gh not found"**
   - Install GitHub CLI: `brew install gh` or follow [installation guide](https://cli.github.com/manual/installation)

2. **"twine upload failed"**
   - Check PyPI credentials in `~/.pypirc`
   - Ensure API tokens are valid
   - Try TestPyPI first: `./publish.sh` (without --prod)

3. **"Failed to build documentation"**
   - Install docs dependencies: `pip install -e ".[docs]"`
   - Check for Sphinx errors in the output

4. **"Tag already exists"**
   - Update version in `pyproject.toml`
   - Or delete existing tag: `git tag -d v0.0.1a3 && git push origin :refs/tags/v0.0.1a3`

5. **"GitHub CLI not authenticated"**
   - Run: `gh auth login` and follow prompts

### Manual Steps (if script fails)

If the automated script fails, you can run steps manually:

```bash
# 1. Build docs
make -C docs html
ghp-import -n -p docs/build/html

# 2. Build package
python -m build

# 3. Upload to TestPyPI
twine upload --repository testpypi dist/*

# 4. Create tag and release
git tag v0.0.1a4
git push origin v0.0.1a4
gh release create v0.0.1a4 --notes "Release v0.0.1a4" dist/*
```

## Best Practices

1. **Always test on TestPyPI first** before production
2. **Update CHANGELOG.md** for every release
3. **Use semantic versioning** (e.g., 1.0.0, 1.1.0, 1.1.1)
4. **Test the package installation** after publishing
5. **Keep release notes clear and informative**

## Support

If you encounter issues with the publishing process:

1. Check this guide for common solutions
2. Review the script output for specific error messages
3. Test individual steps manually
4. Create an issue in the repository if needed