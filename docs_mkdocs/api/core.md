# Core API Reference

This page documents the core `grcwa.obj` class and its methods.

## Class: `grcwa.obj`

The main class for RCWA simulations.

### Constructor

```python
grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
```

Creates an RCWA simulation object for a periodic photonic structure.

**Parameters:**

- **nG** (`int`): Target truncation order for Fourier expansion
    - Actual `nG` may be adjusted based on truncation scheme
    - Typical values: 51-301
    - Higher = more accurate but slower

- **L1** (`list[float, float]`): First lattice vector `[Lx1, Ly1]`
    - Defines periodicity in first direction
    - Units: your chosen length unit (μm, nm, etc.)
    - Example: `[1.0, 0]` for x-direction period of 1.0

- **L2** (`list[float, float]`): Second lattice vector `[Lx2, Ly2]`
    - Defines periodicity in second direction
    - Need not be orthogonal to L1
    - Example: `[0, 1.0]` for square lattice

- **freq** (`float`): Operating frequency
    - In natural units: $f = 1/\lambda$
    - Units: 1/(length unit)
    - Example: `freq=1.0` means $\lambda=1.0$ in your units

- **theta** (`float`): Polar incident angle (radians)
    - Angle from z-axis (surface normal)
    - Range: $[0, \pi/2]$ typically
    - Normal incidence: `theta=0`

- **phi** (`float`): Azimuthal incident angle (radians)
    - Angle in xy-plane from x-axis
    - Range: $[0, 2\pi]$
    - For normal incidence, arbitrary

- **verbose** (`int`, optional): Verbosity level, default=1
    - `0`: Silent
    - `1`: Normal output
    - `2`: Debug information

**Attributes:**

After initialization:

- `obj.nG` (`int`): Actual truncation order used
- `obj.omega` (`complex`): Angular frequency $2\pi f$
- `obj.Layer_N` (`int`): Total number of layers (after adding)
- `obj.G` (`ndarray`): Reciprocal lattice indices `(nG, 2)`
- `obj.kx` (`ndarray`): x-components of wave vectors `(nG,)`
- `obj.ky` (`ndarray`): y-components of wave vectors `(nG,)`

**Example:**

```python
import grcwa
import numpy as np

# Square lattice, wavelength=1.5, normal incidence
obj = grcwa.obj(nG=101,
                L1=[1.5, 0],
                L2=[0, 1.5],
                freq=1.0/1.5,  # λ=1.5
                theta=0,
                phi=0,
                verbose=1)

print(f"Actual nG: {obj.nG}")
print(f"Angular frequency: {obj.omega}")
```

---

## Layer Methods

### `Add_LayerUniform()`

Add a uniform (homogeneous) dielectric layer.

```python
obj.Add_LayerUniform(thickness, epsilon)
```

**Parameters:**

- **thickness** (`float`): Layer thickness in length units
    - Must be positive
    - Can be very large for semi-infinite approximation

- **epsilon** (`float` or `complex`): Relative permittivity $\varepsilon_r$
    - Real: lossless dielectric
    - Complex: lossy/absorbing medium
    - For metals, use complex permittivity or Drude model
    - Example: Silicon $\varepsilon = 12.1$, $n = 3.48$

**Returns:** None

**Example:**

```python
# Vacuum layer
obj.Add_LayerUniform(1.0, 1.0)

# Silicon layer (n=3.48, ε=n²=12.1)
obj.Add_LayerUniform(0.5, 12.1)

# Lossy dielectric
obj.Add_LayerUniform(0.3, 4.0 + 0.1j)

# Metal (Drude model)
eps_inf = 1.0
omega_p = 9.0
gamma = 0.05
eps_metal = eps_inf - omega_p**2 / (obj.omega**2 + 1j*obj.omega*gamma)
obj.Add_LayerUniform(0.05, eps_metal)
```

---

### `Add_LayerGrid()`

Add a patterned layer defined on a Cartesian grid.

```python
obj.Add_LayerGrid(thickness, Nx, Ny)
```

**Parameters:**

- **thickness** (`float`): Layer thickness
- **Nx** (`int`): Number of grid points in x-direction
    - Typical: 100-500
    - Higher = more accurate pattern representation
- **Ny** (`int`): Number of grid points in y-direction

**Returns:** None

**Notes:**

- Pattern must be provided later using `GridLayer_geteps()`
- Grid covers one unit cell: $x, y \in [0, 1]$ (normalized)
- Higher resolution better for sharp features
- Computation time scales with `Nx*Ny` for FFT

**Example:**

```python
# Add patterned layer with 400x400 grid
obj.Add_LayerGrid(thickness=0.3, Nx=400, Ny=400)
```

Later, after all layers added:

```python
# Create pattern
Nx, Ny = 400, 400
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Silicon with circular air hole
eps = np.ones((Nx, Ny)) * 12.1
hole = (X-0.5)**2 + (Y-0.5)**2 < 0.4**2
eps[hole] = 1.0

# Input pattern
obj.GridLayer_geteps(eps.flatten())
```

---

### `Add_LayerFourier()`

Add a patterned layer defined by analytical Fourier coefficients.

```python
obj.Add_LayerFourier(thickness, params)
```

**Parameters:**

- **thickness** (`float`): Layer thickness
- **params**: Parameters defining Fourier series
    - Format depends on implementation
    - Used for shapes with known Fourier series (circles, rectangles)

**Returns:** None

**Note:** This method is available but not commonly used. Grid method is more flexible.

---

## Initialization Method

### `Init_Setup()`

Initialize reciprocal lattice and compute eigenvalues for uniform layers.

```python
obj.Init_Setup(Pscale=1.0, Gmethod=0)
```

**Parameters:**

- **Pscale** (`float`, optional): Period scaling factor, default=1.0
    - Scales lattice vectors: $\mathbf{L}_i \to P_{\text{scale}} \cdot \mathbf{L}_i$
    - Useful for period sweeps with autograd

- **Gmethod** (`int`, optional): Truncation scheme, default=0
    - `0`: Circular truncation (isotropic)
    - `1`: Rectangular/parallelogramic truncation

**Returns:** None

**What it does:**

1. Computes reciprocal lattice vectors $\mathbf{K}_1, \mathbf{K}_2$
2. Generates set of reciprocal lattice points $\mathbf{G}_{mn}$
3. Computes wave vectors $\mathbf{k}_{mn}$ for all diffraction orders
4. Solves eigenvalue problems for uniform layers

**Must be called before:**

- `GridLayer_geteps()`
- `MakeExcitationPlanewave()`
- `RT_Solve()`

**Example:**

```python
obj.Add_LayerUniform(1.0, 1.0)
obj.Add_LayerGrid(0.5, 200, 200)
obj.Add_LayerUniform(1.0, 1.0)

# Initialize with circular truncation
obj.Init_Setup(Gmethod=0)

print(f"Reciprocal vectors: K1={obj.Lk1}, K2={obj.Lk2}")
print(f"Number of orders: {obj.nG}")
```

---

## Pattern Input Method

### `GridLayer_geteps()`

Input dielectric pattern(s) for grid-based patterned layer(s).

```python
obj.GridLayer_geteps(ep_all)
```

**Parameters:**

- **ep_all** (`ndarray`): Flattened array of dielectric constants
    - For 1 patterned layer: shape `(Nx*Ny,)`
    - For N patterned layers: shape `(Nx1*Ny1 + Nx2*Ny2 + ... + NxN*NyN,)`
    - Must be flattened in C-order (row-major)
    - Can be real or complex

**Returns:** None

**Example (single layer):**

```python
obj.Add_LayerGrid(0.3, 100, 100)
obj.Init_Setup()

# Create pattern
eps_grid = np.ones((100, 100)) * 4.0
# ... modify eps_grid ...

# Input pattern
obj.GridLayer_geteps(eps_grid.flatten())
```

**Example (multiple layers):**

```python
obj.Add_LayerGrid(0.3, 100, 100)  # Layer 1
obj.Add_LayerGrid(0.4, 150, 150)  # Layer 2
obj.Init_Setup()

eps1 = np.ones((100, 100)) * 4.0
eps2 = np.ones((150, 150)) * 6.0
# ... modify patterns ...

# Concatenate and input
eps_all = np.concatenate([eps1.flatten(), eps2.flatten()])
obj.GridLayer_geteps(eps_all)
```

---

## Excitation Method

### `MakeExcitationPlanewave()`

Define plane wave excitation.

```python
obj.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order=0, direction=0)
```

**Parameters:**

- **p_amp** (`float`): P-polarization amplitude
    - P = TM = electric field in plane of incidence
    - Typically 0 or 1

- **p_phase** (`float`): P-polarization phase (radians)
    - Usually 0

- **s_amp** (`float`): S-polarization amplitude
    - S = TE = electric field perpendicular to plane of incidence
    - Typically 0 or 1

- **s_phase** (`float`): S-polarization phase (radians)
    - Use ±π/2 for circular polarization

- **order** (`int`, optional): Diffraction order index, default=0
    - Usually 0 (normal plane wave incidence)
    - For oblique incidence from specific order: use other values

- **direction** (`int`, optional): Incidence direction, default=0
    - `0`: Incident from top (input region)
    - `1`: Incident from bottom (output region)

**Returns:** None

**Common polarizations:**

```python
# P-polarized
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=0, s_phase=0, order=0)

# S-polarized
obj.MakeExcitationPlanewave(p_amp=0, p_phase=0,
                             s_amp=1, s_phase=0, order=0)

# 45° linear polarization
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=0, order=0)

# Left circular polarization (LCP)
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=np.pi/2, order=0)

# Right circular polarization (RCP)
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=1, s_phase=-np.pi/2, order=0)
```

---

## Solver Methods

### `RT_Solve()`

Compute reflection and transmission.

```python
R, T = obj.RT_Solve(normalize=0, byorder=0)
```

**Parameters:**

- **normalize** (`int`, optional): Normalization mode, default=0
    - `0`: Raw power (not normalized)
    - `1`: Normalized by incident power and medium properties

- **byorder** (`int`, optional): Output mode, default=0
    - `0`: Total R and T (scalars)
    - `1`: Per-order R and T (arrays of length `nG`)

**Returns:**

- If `byorder=0`: `(R, T)` where R, T are floats
- If `byorder=1`: `(Ri, Ti)` where Ri, Ti are arrays of length `nG`

**Notes:**

- Use `normalize=1` for physically meaningful results
- For lossless structures: $R + T = 1$ (energy conservation)
- For lossy structures: $R + T < 1$ (absorption)

**Example:**

```python
# Total reflection and transmission
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")

# By diffraction order
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
print(f"0th order reflection: {Ri[0]:.4f}")
print(f"0th order transmission: {Ti[0]:.4f}")
print(f"Higher orders: {sum(Ri[1:]) + sum(Ti[1:]):.4f}")

# Check which orders propagate
for i in range(obj.nG):
    if Ri[i] > 1e-6 or Ti[i] > 1e-6:
        m, n = obj.G[i]
        print(f"Order ({m:2d},{n:2d}): R={Ri[i]:.4f}, T={Ti[i]:.4f}")
```

---

## Field Methods

See [Field Methods](fields.md) for detailed documentation of:

- `GetAmplitudes()`: Get mode amplitudes
- `Solve_FieldFourier()`: Compute fields in Fourier space
- `Solve_FieldOnGrid()`: Compute fields in real space
- `Volume_integral()`: Compute volume integrals
- `Solve_ZStressTensorIntegral()`: Compute Maxwell stress tensor

---

## Utility Methods

### `Return_eps()`

Reconstruct dielectric profile from Fourier series.

```python
eps_recon = obj.Return_eps(which_layer, Nx, Ny, component=0)
```

**Parameters:**

- **which_layer** (`int`): Layer index
- **Nx**, **Ny** (`int`): Grid size for reconstruction
- **component** (`int`, optional): Tensor component, default=0
    - `0`: εxx (or scalar ε)
    - `1`: εyy
    - `2`: εzz

**Returns:**

- **eps_recon** (`ndarray`): Reconstructed ε, shape `(Nx, Ny)`

**Example:**

```python
# Reconstruct pattern in layer 1
eps_recon = obj.Return_eps(which_layer=1, Nx=200, Ny=200)

import matplotlib.pyplot as plt
plt.imshow(eps_recon.T, origin='lower')
plt.colorbar(label='ε')
plt.title('Reconstructed Dielectric Pattern')
plt.show()
```

---

## Backend Configuration

### `set_backend()`

Switch computational backend.

```python
grcwa.set_backend(backend_name)
```

**Parameters:**

- **backend_name** (`str`): Backend to use
    - `'numpy'`: Standard NumPy (faster, no gradients)
    - `'autograd'`: Autograd-compatible (slower, supports gradients)

**Example:**

```python
import grcwa

# Use autograd for optimization
grcwa.set_backend('autograd')
import autograd.numpy as np
from autograd import grad

# Define objective
def objective(epsilon):
    obj = grcwa.obj(...)
    # ... setup ...
    obj.GridLayer_geteps(epsilon.flatten())
    R, T = obj.RT_Solve(normalize=1)
    return -R  # Maximize reflection

# Compute gradient
grad_obj = grad(objective)
```

**Note:** Backend must be set before importing grcwa in the same script, or use `importlib.reload()`.

---

## Complete Example

```python
import grcwa
import numpy as np

# Setup
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.0
nG = 101

obj = grcwa.obj(nG, L1, L2, freq, theta=0, phi=0, verbose=1)

# Add layers
obj.Add_LayerUniform(1.0, 1.0)        # vacuum
obj.Add_LayerGrid(0.3, 200, 200)      # patterned
obj.Add_LayerUniform(1.0, 1.0)        # vacuum

# Initialize
obj.Init_Setup()

# Create pattern
Nx, Ny = 200, 200
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

eps = np.ones((Nx, Ny)) * 12.0
hole = (X-0.5)**2 + (Y-0.5)**2 < 0.3**2
eps[hole] = 1.0

obj.GridLayer_geteps(eps.flatten())

# Excitation
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                             s_amp=0, s_phase=0, order=0)

# Solve
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")

# Get fields
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(1, 0.15, [100, 100])
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

# Plot
import matplotlib.pyplot as plt
plt.imshow(I.T, origin='lower', cmap='hot')
plt.colorbar(label='Intensity')
plt.title('Field Intensity in Patterned Layer')
plt.show()
```

---

## See Also

- [Layer Methods](layers.md): Detailed layer management
- [Solver Methods](solver.md): Advanced solving options
- [Field Methods](fields.md): Field analysis and visualization
- [Tutorials](../tutorials/tutorial1.md): Step-by-step examples
