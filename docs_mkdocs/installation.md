# Installation Guide

## Requirements

GRCWA requires Python 3.5 or later. The core dependencies are:

- **numpy**: Numerical computing
- **autograd**: Automatic differentiation

Optional dependencies:

- **nlopt**: For optimization examples (topology optimization)
- **matplotlib**: For visualization
- **pytest**: For running tests

## Installation Methods

### Method 1: Install from PyPI (Recommended)

The simplest way to install GRCWA is via pip:

```bash
pip install grcwa
```

This will automatically install numpy and autograd.

### Method 2: Install from Source

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/weiliangjinca/grcwa.git
cd grcwa

# Install in development mode
pip install -e .
```

Development mode (`-e`) allows you to modify the source code and have changes reflected immediately.

### Method 3: Install with Optional Dependencies

To install with optimization support:

```bash
pip install grcwa nlopt matplotlib
```

## Verify Installation

Test your installation:

```python
import grcwa
import numpy as np

print(f"GRCWA imported successfully!")
print(f"NumPy version: {np.__version__}")

# Test backend
grcwa.set_backend('numpy')
print("NumPy backend: OK")

grcwa.set_backend('autograd')
print("Autograd backend: OK")
```

Run a quick simulation:

```python
import grcwa
import numpy as np

# Simple 3-layer structure
obj = grcwa.obj(nG=101, L1=[1,0], L2=[0,1],
                freq=1.0, theta=0, phi=0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)  # vacuum
obj.Add_LayerUniform(0.5, 4.0)  # dielectric slab
obj.Add_LayerUniform(1.0, 1.0)  # vacuum
obj.Init_Setup()

# Excitation
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0, order=0)

# Solve
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")
```

Expected output:
```
R = 0.3600, T = 0.6400, R+T = 1.0000
```

## Setting Up Development Environment

If you plan to contribute or modify GRCWA:

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv grcwa_env
source grcwa_env/bin/activate  # On Windows: grcwa_env\Scripts\activate

# Using conda
conda create -n grcwa python=3.8
conda activate grcwa
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

Or manually:

```bash
pip install pytest pytest-cov flake8 sphinx
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_rcwa.py::test_rcwa

# Run with coverage
pytest --cov=grcwa tests/
```

## Backend Configuration

GRCWA supports two computational backends:

### NumPy Backend (Default)

Standard NumPy operations - fast, but no automatic differentiation:

```python
import grcwa
grcwa.set_backend('numpy')
```

Use this when:
- You only need forward simulation (R, T, fields)
- Maximum performance is required
- Gradients are not needed

### Autograd Backend

Autograd-compatible NumPy - enables automatic differentiation:

```python
import grcwa
grcwa.set_backend('autograd')
```

Use this when:
- You need gradients for optimization
- Performing inverse design
- Computing parameter sensitivities

!!! warning "Backend Compatibility"
    Once you switch to autograd backend, make sure all your arrays are created with autograd.numpy, not standard numpy:

    ```python
    import autograd.numpy as np  # Use this
    # not: import numpy as np
    ```

## Common Installation Issues

### Issue: "No module named autograd"

**Solution**: Install autograd:
```bash
pip install autograd
```

### Issue: Numpy version conflict

**Solution**: Ensure compatible numpy version:
```bash
pip install --upgrade numpy autograd
```

### Issue: "ImportError: cannot import name 'obj'"

**Solution**: Make sure you're importing from the package:
```python
import grcwa
obj = grcwa.obj(...)  # Correct

# Not:
from grcwa import obj  # May not work in older versions
```

### Issue: Tests fail with "ModuleNotFoundError"

**Solution**: Install test dependencies:
```bash
pip install pytest
```

### Issue: NLOPT not available

NLOPT is optional and only needed for optimization examples.

**Linux**:
```bash
pip install nlopt
```

**Mac** (with Homebrew):
```bash
brew install nlopt
pip install nlopt
```

**Windows**: Use conda:
```bash
conda install -c conda-forge nlopt
```

## Docker Installation (Optional)

For a reproducible environment:

Create `Dockerfile`:
```dockerfile
FROM python:3.8-slim

RUN pip install grcwa nlopt matplotlib jupyter

WORKDIR /workspace

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
```

Build and run:
```bash
docker build -t grcwa .
docker run -p 8888:8888 -v $(pwd):/workspace grcwa
```

## Platform-Specific Notes

### Linux
Should work out of the box with pip.

### macOS
May need Xcode command line tools:
```bash
xcode-select --install
```

### Windows
Works with standard Python installation. If using Anaconda:
```bash
conda install numpy autograd
pip install grcwa
```

## Performance Optimization

### NumPy with MKL

For better performance, use NumPy built with Intel MKL:

```bash
# With conda
conda install numpy scipy mkl

# Verify
python -c "import numpy; numpy.show_config()"
```

### Parallel Computing

GRCWA itself doesn't use multi-threading, but you can parallelize parameter sweeps:

```python
from multiprocessing import Pool

def compute_spectrum(freq):
    obj = grcwa.obj(...)
    # ... setup ...
    R, T = obj.RT_Solve()
    return R, T

freqs = np.linspace(0.5, 1.5, 50)
with Pool(8) as p:
    results = p.map(compute_spectrum, freqs)
```

## Next Steps

Now that GRCWA is installed:

1. **[Quick Start Guide](quickstart.md)**: Run your first simulation
2. **[Basic Concepts](guide/concepts.md)**: Understand the workflow
3. **[Examples](examples/gallery.md)**: Explore example simulations
4. **[Tutorials](tutorials/tutorial1.md)**: Step-by-step learning

## Getting Help

If you encounter problems:

- Check the [Troubleshooting](reference/troubleshooting.md) guide
- Read the [FAQ](reference/faq.md)
- Open an issue on [GitHub](https://github.com/weiliangjinca/grcwa/issues)
- Contact: jwlaaa@gmail.com
