# Variables and Conventions Reference

This page documents all important variables, their meanings, and conventions used in GRCWA.

## Physical Constants and Units

### Natural Units

GRCWA uses **natural units** where:

| Constant | Value | Meaning |
|----------|-------|---------|
| $\varepsilon_0$ | 1 | Vacuum permittivity |
| $\mu_0$ | 1 | Vacuum permeability |
| $c$ | 1 | Speed of light |
| $Z_0 = \sqrt{\mu_0/\varepsilon_0}$ | 1 | Vacuum impedance |

**Implications:**

- Choose any length unit (μm, nm, mm, etc.)
- Frequency $f = 1/\lambda$ in your chosen units
- Dielectric constants are dimensionless: $\varepsilon_r = \varepsilon/\varepsilon_0$
- All equations simplified by $c=1$

### Example: Wavelength 1.55 μm

```python
wavelength = 1.55  # μm
freq = 1.0 / wavelength  # freq ≈ 0.645 in natural units
L1 = [0.5, 0]  # Lattice constant 0.5 μm
thickness = 0.3  # Layer thickness 0.3 μm
```

All lengths must use the **same unit** (μm in this example).

## Time Harmonic Convention

Fields oscillate as:

$$
\mathbf{E}(\mathbf{r}, t) = \text{Re}[\mathbf{E}(\mathbf{r}) e^{-i\omega t}]
$$

**Convention**: $e^{-i\omega t}$ (not $e^{+i\omega t}$)

**Implications:**

- Phase advance in space: $e^{+ikz}$ for forward propagation
- Absorption: $\varepsilon = \varepsilon' + i\varepsilon''$ with $\varepsilon'' > 0$
- Evanescent decay: $e^{-\kappa z}$ with $\kappa > 0$

## Core Class: `grcwa.obj`

### Constructor Parameters

```python
obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `nG` | int | - | Target number of Fourier harmonics |
| `L1` | list[float, float] | length | Lattice vector 1: `[Lx1, Ly1]` |
| `L2` | list[float, float] | length | Lattice vector 2: `[Lx2, Ly2]` |
| `freq` | float | 1/length | Frequency ($\omega/(2\pi c)$) |
| `theta` | float | radians | Polar incident angle (from z-axis) |
| `phi` | float | radians | Azimuthal angle (in xy-plane from x-axis) |
| `verbose` | int | - | Verbosity level: 0 (quiet), 1 (normal), 2 (debug) |

**Notes:**

- Actual `nG` may differ due to truncation scheme
- `L1`, `L2` need not be orthogonal (supports oblique lattices)
- Normal incidence: `theta=0`, `phi=0`
- Oblique incidence: $0 < \theta < \pi/2$

### Lattice Vectors

```python
L1 = [Lx1, Ly1]  # First lattice vector
L2 = [Lx2, Ly2]  # Second lattice vector
```

**Common lattices:**

**Square:**
```python
a = 1.0
L1 = [a, 0]
L2 = [0, a]
```

**Rectangular:**
```python
a, b = 1.0, 0.5
L1 = [a, 0]
L2 = [0, b]
```

**Hexagonal:**
```python
a = 1.0
L1 = [a, 0]
L2 = [a/2, a*np.sqrt(3)/2]  # 60° angle
```

**Rhombohedral:**
```python
a = 1.0
angle = 75 * np.pi/180
L1 = [a, 0]
L2 = [a*np.cos(angle), a*np.sin(angle)]
```

### Angles

**Theta** ($\theta$): Polar angle from z-axis

- $\theta = 0$: Normal incidence
- $0 < \theta < \pi/2$: Oblique incidence from above
- $\pi/2 < \theta < \pi$: Grazing incidence (rarely used)

**Phi** ($\phi$): Azimuthal angle in xy-plane

- $\phi = 0$: Incident in xz-plane
- $\phi = \pi/2$: Incident in yz-plane
- General: Incident direction is $(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$

**Wave vector:**

$$
\mathbf{k}_{\text{inc}} = \omega\sqrt{\varepsilon_{\text{in}}}(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)
$$

## Layer Variables

### Layer Types

| Type ID | Method | Description |
|---------|--------|-------------|
| 0 | `Add_LayerUniform` | Homogeneous dielectric |
| 1 | `Add_LayerGrid` | Pattern defined on Cartesian grid |
| 2 | `Add_LayerFourier` | Pattern defined by Fourier coefficients |

### Uniform Layer

```python
obj.Add_LayerUniform(thickness, epsilon)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `thickness` | float | length | Layer thickness |
| `epsilon` | float or complex | - | Relative permittivity $\varepsilon_r$ |

**Examples:**

```python
# Vacuum
obj.Add_LayerUniform(1.0, 1.0)

# Silicon (n=3.48)
obj.Add_LayerUniform(0.5, 3.48**2)  # ε = n²

# Silicon with loss
obj.Add_LayerUniform(0.5, 12.1 + 0.1j)  # ε = ε' + iε''

# Silver (Drude model)
eps_inf = 5.0
omega_p = 9.0  # Plasma frequency
gamma = 0.02  # Damping
eps_Ag = eps_inf - omega_p**2 / (freq**2 + 1j*freq*gamma)
obj.Add_LayerUniform(0.02, eps_Ag)
```

### Patterned Layer (Grid)

```python
obj.Add_LayerGrid(thickness, Nx, Ny)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `thickness` | float | length | Layer thickness |
| `Nx` | int | - | Number of grid points in x |
| `Ny` | int | - | Number of grid points in y |

Later, input pattern:

```python
epsilon_grid = ...  # Shape: (Nx, Ny)
obj.GridLayer_geteps(epsilon_grid.flatten())
```

**Pattern coordinates:**

- Grid points at $(i/(N_x-1), j/(N_y-1))$ for $i=0,\ldots,N_x-1$, $j=0,\ldots,N_y-1$
- Normalized coordinates $x, y \in [0, 1]$
- Physical coordinates: multiply by lattice vectors

**Multiple patterned layers:**

If you have multiple patterned layers, flatten and concatenate:

```python
epsilon1 = ...  # (Nx1, Ny1)
epsilon2 = ...  # (Nx2, Ny2)
epsilon_all = np.concatenate([epsilon1.flatten(), epsilon2.flatten()])
obj.GridLayer_geteps(epsilon_all)
```

### Grid Resolution

**Recommendations:**

| Feature Size | Recommended Nx, Ny |
|--------------|-------------------|
| Smooth variations | 50-100 |
| Sharp features | 200-400 |
| Very fine details | 500-1000 |

**Trade-off:** Higher resolution → more accurate but slower FFT.

## Fourier Harmonics

### Truncation Order

```python
nG = 101  # Target number
```

After initialization, actual `nG`:

```python
obj.nG  # Actual number used
obj.G   # Array of (m, n) indices, shape: (nG, 2)
```

**Truncation schemes:**

- **Circular** (`Gmethod=0`, default): $m^2 + n^2 \leq N_{\max}^2$
- **Rectangular** (`Gmethod=1`): $|m| \leq M$, $|n| \leq N$

**Convergence:**

```python
# Test convergence
nG_values = [51, 101, 201, 301, 501]
for nG in nG_values:
    obj = grcwa.obj(nG, ...)
    # ... solve ...
    R, T = obj.RT_Solve()
    print(f"nG={obj.nG}, R={R:.6f}, T={T:.6f}")
```

Converged when R, T don't change significantly with increasing `nG`.

## Reciprocal Lattice

After `Init_Setup()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `obj.Lk1` | array | Reciprocal lattice vector 1 |
| `obj.Lk2` | array | Reciprocal lattice vector 2 |
| `obj.G` | array (nG, 2) | Array of $(m, n)$ indices |
| `obj.kx` | array (nG,) | x-components of wave vectors |
| `obj.ky` | array (nG,) | y-components of wave vectors |

**Reciprocal lattice vectors:**

$$
\mathbf{K}_1 \cdot \mathbf{L}_1 = 2\pi, \quad \mathbf{K}_1 \cdot \mathbf{L}_2 = 0
$$

$$
\mathbf{K}_2 \cdot \mathbf{L}_1 = 0, \quad \mathbf{K}_2 \cdot \mathbf{L}_2 = 2\pi
$$

**Wave vectors for each order:**

$$
k_{x,mn} = k_{x0} + m K_{1x} + n K_{2x}
$$

$$
k_{y,mn} = k_{y0} + m K_{1y} + n K_{2y}
$$

Access in code:

```python
kx_mn = obj.kx  # Shape: (nG,)
ky_mn = obj.ky  # Shape: (nG,)
G_indices = obj.G  # Shape: (nG, 2), each row is (m, n)
```

## Excitation Variables

### Plane Wave Excitation

```python
obj.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order, direction=0)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `p_amp` | float | - | P-polarization amplitude |
| `p_phase` | float | radians | P-polarization phase |
| `s_amp` | float | - | S-polarization amplitude |
| `s_phase` | float | radians | S-polarization phase |
| `order` | int | - | Diffraction order index (usually 0) |
| `direction` | int | - | 0: from top, 1: from bottom |

**Polarization definitions:**

**P-polarization (TM)**: E-field in plane of incidence

$$
\hat{p} = \frac{\mathbf{k}_\parallel \times \hat{z} \times \mathbf{k}}{|\mathbf{k}_\parallel \times \hat{z} \times \mathbf{k}|}
$$

**S-polarization (TE)**: E-field perpendicular to plane of incidence

$$
\hat{s} = \frac{\mathbf{k} \times \hat{z}}{|\mathbf{k} \times \hat{z}|}
$$

**Common cases:**

```python
# P-polarized
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

# S-polarized
obj.MakeExcitationPlanewave(0, 0, 1, 0, 0)

# 45° linear polarization
obj.MakeExcitationPlanewave(1, 0, 1, 0, 0)

# Left circular polarization (LCP)
obj.MakeExcitationPlanewave(1, 0, 1, np.pi/2, 0)

# Right circular polarization (RCP)
obj.MakeExcitationPlanewave(1, 0, 1, -np.pi/2, 0)
```

## Solution Variables

### Reflection and Transmission

```python
R, T = obj.RT_Solve(normalize=1, byorder=0)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `normalize` | int | 0: raw, 1: normalized |
| `byorder` | int | 0: total, 1: by order |

**Returns:**

- `normalize=1, byorder=0`: $(R, T)$ scalars, total power
- `normalize=1, byorder=1`: $(R_i, T_i)$ arrays of length `nG`

**Normalization:**

With `normalize=1`:

$$
R = \sum_{mn} \frac{\text{Re}(k_{z,mn}^{\text{refl}})}{\text{Re}(k_{z,\text{inc}})} |r_{mn}|^2
$$

$$
T = \sum_{mn} \frac{\text{Re}(k_{z,mn}^{\text{trans}}) \varepsilon_{\text{in}}}{\text{Re}(k_{z,\text{inc}}) \varepsilon_{\text{out}}} |t_{mn}|^2
$$

Energy conservation: $R + T = 1$ for lossless structures.

### Field Amplitudes

```python
a_i, b_i = obj.GetAmplitudes(which_layer, z_offset)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `which_layer` | int | Layer index (0-based) |
| `z_offset` | float | Position within layer |

**Returns:**

- `a_i`: Forward mode amplitudes, shape `(2*nG,)` for $(E_x, E_y)$ components
- `b_i`: Backward mode amplitudes, shape `(2*nG,)`

Format: `[Ex_0, Ex_1, ..., Ex_nG, Ey_0, Ey_1, ..., Ey_nG]`

### Fields in Fourier Space

```python
[Ex_mn, Ey_mn, Ez_mn], [Hx_mn, Hy_mn, Hz_mn] = obj.Solve_FieldFourier(which_layer, z_offset)
```

**Returns:** 6 arrays, each shape `(nG,)`, complex values

- Fourier coefficients for each component
- To get total field: sum over all orders with phase factors

### Fields in Real Space

```python
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(which_layer, z_offset, Nxy)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `Nxy` | list[int, int] | Grid size `[Nx, Ny]` |

**Returns:** 6 arrays, each shape `(Nx, Ny)`, complex values

- Electric and magnetic field components on real-space grid
- Intensity: $I = |E_x|^2 + |E_y|^2 + |E_z|^2$

## Internal Variables

### Layer Storage

| Attribute | Description |
|-----------|-------------|
| `obj.Layer_N` | Total number of layers |
| `obj.thickness_list` | List of layer thicknesses |
| `obj.id_list` | Layer type identifiers |
| `obj.Uniform_ep_list` | Dielectric constants for uniform layers |
| `obj.GridLayer_N` | Number of grid-based layers |
| `obj.GridLayer_Nxy_list` | Grid sizes for each patterned layer |

### Eigenmode Variables

| Attribute | Description |
|-----------|-------------|
| `obj.q_list` | Eigenvalues $q$ for each layer |
| `obj.phi_list` | Eigenvectors for each layer |
| `obj.kp_list` | $K_\perp$ matrices for each layer |

These are computed by `Init_Setup()` and used internally for solving.

## Common Combinations

### Spectral Sweep

```python
wavelengths = np.linspace(0.4, 0.8, 100)
R_spectrum = []

for wl in wavelengths:
    freq = 1.0 / wl
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    # ... add layers, solve ...
    R, T = obj.RT_Solve(normalize=1)
    R_spectrum.append(R)
```

### Angle Sweep

```python
angles = np.linspace(0, 80, 50) * np.pi/180
R_angle = []

for theta in angles:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi=0, verbose=0)
    # ... add layers, solve ...
    R, T = obj.RT_Solve(normalize=1)
    R_angle.append(R)
```

### Polarization Sweep

```python
pols = np.linspace(0, np.pi, 90)  # Polarization angle
R_pol = []

for pol in pols:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    # ... add layers ...
    obj.MakeExcitationPlanewave(np.cos(pol), 0, np.sin(pol), 0, 0)
    R, T = obj.RT_Solve(normalize=1)
    R_pol.append(R)
```

## Autograd Variables

When using `grcwa.set_backend('autograd')`:

```python
import autograd.numpy as np
from autograd import grad

# Autogradable parameters
epsilon_grid = np.array(...)  # Must use autograd.numpy
freq = np.array(1.0)
theta = np.array(0.1)
thickness = np.array(0.5)

# Compute gradient
def objective(eps):
    # ... setup and solve ...
    R, T = obj.RT_Solve()
    return -R  # Maximize R

grad_obj = grad(objective)
gradient = grad_obj(epsilon_grid)
```

**Autogradable parameters:**

- ✅ Dielectric values on grid
- ✅ Frequency `freq`
- ✅ Angles `theta`, `phi`
- ✅ Layer thickness
- ✅ Periodicity scaling `Pscale`
- ❌ Truncation order `nG`
- ❌ Grid sizes `Nx`, `Ny`
- ❌ Layer numbers

## Summary Table

| Variable | Symbol | Type | Units | Description |
|----------|--------|------|-------|-------------|
| Frequency | $\omega/(2\pi)$ | float | 1/length | Operating frequency |
| Wavelength | $\lambda$ | float | length | $\lambda = 1/f$ |
| Polar angle | $\theta$ | float | radians | Incident angle from z-axis |
| Azimuthal angle | $\phi$ | float | radians | Angle in xy-plane |
| Lattice vectors | $\mathbf{L}_1, \mathbf{L}_2$ | list | length | Real-space periodicity |
| Reciprocal vectors | $\mathbf{K}_1, \mathbf{K}_2$ | array | 1/length | Fourier-space periodicity |
| Truncation order | $N_G$ | int | - | Number of Fourier harmonics |
| Layer thickness | $d$ | float | length | Thickness of each layer |
| Dielectric constant | $\varepsilon$ | complex | - | Relative permittivity |
| Reflection | $R$ | float | - | Reflected power fraction |
| Transmission | $T$ | float | - | Transmitted power fraction |

## Next Steps

- **[Physical Units](units.md)**: Unit conversion examples
- **[Troubleshooting](troubleshooting.md)**: Common issues and fixes
- **[FAQ](faq.md)**: Frequently asked questions
- **[API Reference](../api/core.md)**: Complete function documentation
