# Frequently Asked Questions (FAQ)

## General Questions

### What is RCWA?

Rigorous Coupled Wave Analysis (RCWA) is a semi-analytical method for solving Maxwell's equations in periodic structures. It expands electromagnetic fields in Fourier series and solves the resulting eigenvalue problems layer by layer. RCWA is exact (within numerical precision) and particularly efficient for periodic photonic structures.

### When should I use RCWA instead of FDTD or FEM?

**Use RCWA when:**

- Your structure has 2D periodicity (photonic crystals, gratings, metasurfaces)
- You need spectral or angular response (frequency/angle sweeps)
- You want fast simulations of periodic structures
- You need far-field diffraction patterns

**Use FDTD/FEM when:**

- Structure is aperiodic (isolated objects, random structures)
- You need time-domain response
- 3D arbitrary geometries without periodicity
- Broadband simulations in single run (FDTD advantage)

### What does "autoGradable" mean?

GRCWA integrates with [Autograd](https://github.com/HIPS/autograd), enabling automatic differentiation. This means you can compute gradients of any output (R, T, fields) with respect to any input (ε, frequency, angles, thickness) automatically, without deriving adjoint equations manually. This is essential for:

- Topology optimization
- Inverse design
- Sensitivity analysis
- Gradient-based optimization

## Installation & Setup

### How do I install GRCWA?

```bash
pip install grcwa
```

For the latest development version:

```bash
git clone https://github.com/weiliangjinca/grcwa
cd grcwa
pip install -e .
```

### What are the dependencies?

**Required:**

- Python ≥ 3.5
- numpy
- autograd

**Optional:**

- nlopt (for optimization examples)
- matplotlib (for visualization)
- pytest (for testing)

### How do I switch between NumPy and Autograd backends?

```python
import grcwa

# NumPy backend (faster, no gradients)
grcwa.set_backend('numpy')

# Autograd backend (gradients enabled)
grcwa.set_backend('autograd')
```

**Important:** Set backend before creating `grcwa.obj` instances.

## Usage Questions

### How do I choose the truncation order (nG)?

**Rules of thumb:**

- Uniform layers: `nG = 11-51`
- Smooth patterns: `nG = 51-101`
- Sharp features: `nG = 101-301`
- Very fine details: `nG = 301-501`

**Always test convergence:**

```python
for nG in [51, 101, 201, 301]:
    obj = grcwa.obj(nG, ...)
    # ... setup and solve ...
    R, T = obj.RT_Solve(normalize=1)
    print(f"nG={obj.nG}: R={R:.6f}, T={T:.6f}")
```

### How many grid points (Nx, Ny) should I use?

**Recommendations:**

- Smooth features: 50-100
- Typical patterns: 200-400
- Sharp edges: 400-500
- Very fine details: 500-1000

**Trade-off:** Higher resolution = more accurate but slower FFT.

### What units should I use?

GRCWA uses **natural units** where $c = \varepsilon_0 = \mu_0 = 1$. You can use any consistent length unit:

```python
# Example: all in μm
wavelength = 1.55  # μm
freq = 1.0 / wavelength  # ≈ 0.645
L1 = [0.5, 0]  # μm
thickness = 0.3  # μm
```

**Key:** All lengths must use the **same unit**.

### How do I convert wavelength to frequency?

In natural units:

$$
f = \frac{c}{\lambda} = \frac{1}{\lambda}
$$

```python
wavelength = 1.5  # Your chosen unit
freq = 1.0 / wavelength
```

### What's the difference between P and S polarization?

**P-polarization (TM):**

- Electric field in the plane of incidence
- Magnetic field perpendicular to plane of incidence
- Set `p_amp=1, s_amp=0`

**S-polarization (TE):**

- Electric field perpendicular to plane of incidence
- Magnetic field in the plane of incidence
- Set `p_amp=0, s_amp=1`

For normal incidence on isotropic materials, P and S give the same result.

### How do I define circular polarization?

**Left circular polarization (LCP):**

```python
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=np.pi/2, order=0)
```

**Right circular polarization (RCP):**

```python
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=-np.pi/2, order=0)
```

## Troubleshooting

### Why is R + T ≠ 1?

**Possible causes:**

1. **Insufficient truncation order:** Increase `nG`
2. **Numerical instability:** Reduce layer thickness or use smaller `nG`
3. **Absorbing material:** For lossy materials, $R + T < 1$ is correct (absorption = $1-R-T$)
4. **Very high contrast:** Use more grid points (`Nx, Ny`)

**Fix:**

```python
# Test convergence
for nG in [101, 201, 301, 501]:
    obj = grcwa.obj(nG, ...)
    # ... solve ...
    R, T = obj.RT_Solve(normalize=1)
    print(f"nG={nG}: R+T={R+T:.6f}, error={(R+T-1):.2e}")
```

### Why am I getting a singular matrix error?

**Causes:**

- Perfect normal incidence with no loss
- Perfectly symmetric structure

**Fix:** Add tiny loss:

```python
Qabs = 1e8  # Very high Q-factor
freq = freq * (1 + 1j / (2*Qabs))
```

Or add tiny imaginary part to epsilon:

```python
epsilon = 4.0 + 1e-10j
```

### My pattern looks wrong. How do I debug?

Use `Return_eps()` to visualize:

```python
eps_recon = obj.Return_eps(which_layer=1, Nx=200, Ny=200)

import matplotlib.pyplot as plt
plt.imshow(eps_recon.T, origin='lower')
plt.colorbar()
plt.title('Reconstructed Dielectric Pattern')
plt.show()
```

Compare with your input pattern to verify correctness.

### Results are very slow. How to speed up?

**Optimization strategies:**

1. **Reduce truncation order:** Try lower `nG` first
2. **Use NumPy backend:** Faster than Autograd if no gradients needed
3. **Reduce grid resolution:** Lower `Nx, Ny` if acceptable
4. **Use uniform layers when possible:** Much faster than patterned
5. **Parallelize parameter sweeps:**

```python
from multiprocessing import Pool

def compute(freq):
    obj = grcwa.obj(...)
    # ... solve ...
    return R, T

with Pool(8) as p:
    results = p.map(compute, frequencies)
```

### How do I handle anisotropic materials?

For anisotropic dielectric tensors, provide a list of 3 components:

```python
# Uniaxial material with εxx = εyy ≠ εzz
eps_xx = 4.0
eps_yy = 4.0
eps_zz = 6.0

eps_tensor = [eps_xx, eps_yy, eps_zz]

# For grid layer
eps_grid_xx = np.ones((Nx, Ny)) * eps_xx
eps_grid_yy = np.ones((Nx, Ny)) * eps_yy
eps_grid_zz = np.ones((Nx, Ny)) * eps_zz

eps_all = [eps_grid_xx.flatten(), eps_grid_yy.flatten(), eps_grid_zz.flatten()]
obj.GridLayer_geteps(eps_all)
```

## Advanced Usage

### How do I do topology optimization?

```python
import grcwa
grcwa.set_backend('autograd')
import autograd.numpy as np
from autograd import grad

def objective(epsilon):
    obj = grcwa.obj(101, [1,0], [0,1], freq, 0, 0, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)
    obj.Add_LayerGrid(0.5, Nx, Ny)
    obj.Add_LayerUniform(1.0, 1.0)
    obj.Init_Setup()
    obj.GridLayer_geteps(epsilon.flatten())
    obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
    R, T = obj.RT_Solve(normalize=1)
    return -R  # Maximize reflection

# Compute gradient
grad_obj = grad(objective)

# Use optimizer (e.g., NLOPT, scipy.optimize)
import nlopt

def nlopt_objective(x, grad_array):
    if grad_array.size > 0:
        grad_array[:] = grad_obj(x)
    return objective(x)

# Setup NLOPT
opt = nlopt.opt(nlopt.LD_MMA, Nx*Ny)
opt.set_min_objective(nlopt_objective)
opt.set_lower_bounds(1.0)
opt.set_upper_bounds(12.0)

# Initial guess
epsilon_init = np.ones(Nx*Ny) * 6.0

# Optimize
epsilon_opt = opt.optimize(epsilon_init)
```

### Can I simulate isolated objects?

RCWA requires periodicity. For isolated objects:

1. **Use large unit cell** (super-cell approach):

```python
# Object size: 1 μm
# Use 10 μm × 10 μm unit cell
L1 = [10, 0]
L2 = [0, 10]

# Place object at center, surrounded by vacuum
```

2. **Check convergence** with increasing cell size until results stabilize

### How do I get reflection/transmission coefficients (not just powers)?

The amplitudes are stored internally. Access via:

```python
# After solving
amplitudes_in, amplitudes_out = obj.GetAmplitudes(layer=0, z_offset=0)
```

For reflection coefficients:

```python
# Reflection coefficients for each order
r_coeffs = amplitudes_out[0:obj.nG]  # Backward propagating in input region
```

### How do I compute absorption?

For lossy materials:

```python
R, T = obj.RT_Solve(normalize=1)
A = 1 - R - T  # Absorption
```

Or use volume integral of $\text{Im}(\varepsilon)|E|^2$:

```python
absorption = obj.Volume_integral(which_layer, Mx, My, Mz, normalize=1)
```

### Can I use GRCWA for 3D photonic crystals?

RCWA assumes periodicity in 2D (xy-plane) and stacking in z. For 3D photonic crystals:

- ✅ **Photonic crystal slabs** (2D periodic, finite in z): Yes
- ❌ **3D bulk photonic crystals** (periodic in xyz): No, use plane-wave expansion or other methods

### How do I cite GRCWA?

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

## Common Error Messages

### "IndexError: index out of range"

**Cause:** Mismatch between number of patterned layers and epsilon arrays provided.

**Fix:** Ensure `epsilon.flatten()` has correct total length.

### "ValueError: operands could not be broadcast together"

**Cause:** Array shape mismatch in pattern definition.

**Fix:** Check that pattern has shape `(Nx, Ny)` and use `indexing='ij'` in meshgrid.

### "LinAlgError: Singular matrix"

**Cause:** Numerical singularity, often at normal incidence with no loss.

**Fix:** Add tiny loss:

```python
freq = freq * (1 + 1e-10j)
```

## Still Have Questions?

- Check the [Troubleshooting Guide](troubleshooting.md)
- Read the [Tutorials](../tutorials/tutorial1.md)
- Browse [Examples](../examples/gallery.md)
- Open an issue on [GitHub](https://github.com/weiliangjinca/grcwa/issues)
- Contact: jwlaaa@gmail.com
