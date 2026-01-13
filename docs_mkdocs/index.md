# Welcome to GRCWA Documentation

<div align="center" markdown>

![RCWA Structure](../imag/scheme.png){ width="600" loading=lazy }

**Autograd-enabled Rigorous Coupled Wave Analysis for Photonic Crystals**

[:octicons-mark-github-16: GitHub](https://github.com/weiliangjinca/grcwa){ .md-button .md-button--primary }
[:fontawesome-brands-python: PyPI](https://pypi.org/project/grcwa/){ .md-button }
[:octicons-book-16: Read the Docs](https://grcwa.readthedocs.io){ .md-button }

</div>

---

## What is GRCWA?

!!! abstract "Overview"
    **GRCWA** (autoGradable Rigorous Coupled Wave Analysis) is a powerful Python library for simulating light interaction with periodic photonic structures. It implements the Rigorous Coupled Wave Analysis (RCWA) method with full support for automatic differentiation, making it ideal for inverse design and optimization of photonic devices.

## :material-features: Key Features

### :microscope: Physics-Based Simulation

<div class="grid cards" markdown>

-   :fontawesome-solid-wave-square: __Rigorous Simulation__

    ---

    Solve Maxwell's equations exactly using Fourier modal method

    - Full vectorial field calculations
    - Arbitrary 2D periodic structures
    - Multiple layer stacking

-   :material-layers-triple: __Multiple Layers__

    ---

    Support complex multilayer structures

    - Independent dielectric profiles
    - Uniform and patterned layers
    - Unlimited layer count

</div>

### :dart: Arbitrary Geometries

=== "Uniform Layers"

    Simple dielectric slabs with constant permittivity

    ```python
    obj.Add_LayerUniform(thickness=0.5, epsilon=4.0)
    ```

=== "Grid-Based Patterns"

    Any 2D pattern with Cartesian grids

    ```python
    obj.Add_LayerGrid(thickness=0.3, Nx=400, Ny=400)
    obj.GridLayer_geteps(epsilon_grid.flatten())
    ```

=== "Analytical Fourier"

    Efficient for shapes with known Fourier series

    ```python
    obj.Add_LayerFourier(thickness=0.2, params)
    ```

### :rocket: Automatic Differentiation

!!! tip "Gradient-Based Optimization"
    Integrated with [Autograd](https://github.com/HIPS/autograd) for automatic gradient computation

**Autogradable Parameters:**

- [x] Dielectric constants at every grid point
- [x] Layer thicknesses
- [x] Operating frequency
- [x] Incident angles
- [x] Lattice periodicity

### :triangular_ruler: Flexible Lattices

<div class="grid cards" markdown>

-   __Square__

    ```python
    L1 = [a, 0]
    L2 = [0, a]
    ```

-   __Hexagonal__

    ```python
    L1 = [a, 0]
    L2 = [a/2, a*√3/2]
    ```

-   __Arbitrary__

    ```python
    L1 = [Lx1, Ly1]
    L2 = [Lx2, Ly2]
    ```

</div>

## What Can You Do with GRCWA?

### Analysis Tasks
- Compute reflection and transmission spectra
- Analyze diffraction orders
- Calculate electromagnetic fields in real and Fourier space
- Compute Poynting flux and energy flow
- Evaluate Maxwell stress tensors

### Design & Optimization
- Topology optimization of photonic structures
- Inverse design of metasurfaces
- Gradient-based optimization of:
    - Optical filters
    - Anti-reflection coatings
    - Photonic band gap structures
    - Broadband reflectors
    - Efficient light absorbers

### Research Applications
- Photonic crystal design
- Metamaterial engineering
- Grating design
- Diffractive optics
- Solar cell optimization
- LIDAR component design

## Quick Example

Here's a simple example to get you started:

```python
import grcwa
import numpy as np

# Define lattice and frequency
L1 = [1.5, 0]  # Lattice vector 1
L2 = [0, 1.5]  # Lattice vector 2
freq = 1.0     # Frequency (c=1)
theta = 0.0    # Incident angle
phi = 0.0      # Azimuthal angle
nG = 101       # Truncation order

# Create RCWA object
obj = grcwa.obj(nG, L1, L2, freq, theta, phi)

# Add layers: vacuum + patterned + vacuum
obj.Add_LayerUniform(1.0, 1.0)        # Vacuum layer
obj.Add_LayerGrid(0.2, 400, 400)       # Patterned layer
obj.Add_LayerUniform(1.0, 1.0)        # Vacuum layer

# Setup reciprocal lattice
obj.Init_Setup()

# Define pattern (circular hole)
Nx, Ny = 400, 400
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
pattern = np.ones((Nx, Ny)) * 4.0  # Silicon (ε=4)
hole = (X-0.5)**2 + (Y-0.5)**2 < 0.3**2
pattern[hole] = 1.0  # Air hole

# Input pattern
obj.GridLayer_geteps(pattern.flatten())

# Setup excitation (p-polarized plane wave)
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0, order=0)

# Solve for reflection and transmission
R, T = obj.RT_Solve(normalize=1)
print(f'R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}')
```

## Why Choose GRCWA?

| Feature | GRCWA | Traditional RCWA |
|---------|-------|------------------|
| Automatic Differentiation | ✅ Built-in | ❌ Manual derivation |
| Optimization Ready | ✅ Direct integration | ❌ Requires external tools |
| Python Native | ✅ Easy to use | ⚠️ Often C/Fortran |
| Arbitrary Patterns | ✅ Grid-based | ⚠️ Limited shapes |
| Active Development | ✅ Open source | ⚠️ Varies |

## Getting Started

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get up and running in minutes with our quick start guide

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } __Learn the Theory__

    ---

    Understand the physics and mathematics behind RCWA

    [:octicons-arrow-right-24: Theory](theory/principles.md)

-   :material-code-braces:{ .lg .middle } __API Reference__

    ---

    Detailed documentation of all classes and functions

    [:octicons-arrow-right-24: API Docs](api/core.md)

-   :material-school:{ .lg .middle } __Tutorials__

    ---

    Step-by-step tutorials for common use cases

    [:octicons-arrow-right-24: Tutorials](tutorials/tutorial1.md)

</div>

## Project Information

- **Author**: Weiliang Jin (jwlaaa@gmail.com)
- **Version**: 0.1.2
- **License**: GPL v3
- **Python**: ≥ 3.5
- **Repository**: [github.com/weiliangjinca/grcwa](https://github.com/weiliangjinca/grcwa)

## Citation

If you use GRCWA in your research, please cite:

```bibtex
@article{Jin2020,
  title = {Inverse design of lightweight broadband reflector for relativistic lightsail propulsion},
  author = {Jin, Weiliang and Li, Wei and Orenstein, Meir and Fan, Shanhui},
  journal = {ACS Photonics},
  volume = {7},
  number = {9},
  pages = {2350--2355},
  year = {2020},
  publisher = {ACS Publications}
}
```

## Need Help?

- 📖 Read the [detailed documentation](introduction.md)
- 💡 Check out [examples](examples/gallery.md)
- 🐛 Report issues on [GitHub](https://github.com/weiliangjinca/grcwa/issues)
- 📧 Contact the author at jwlaaa@gmail.com
