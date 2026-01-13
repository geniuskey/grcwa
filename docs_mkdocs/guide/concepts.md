# Basic Concepts

This guide explains the fundamental concepts needed to use GRCWA effectively.

## The RCWA Workflow

Every RCWA simulation follows these steps:

```python
import grcwa
import numpy as np

# 1. Create RCWA object
obj = grcwa.obj(nG, L1, L2, freq, theta, phi)

# 2. Define layer stack
obj.Add_LayerUniform(thickness1, epsilon1)
obj.Add_LayerGrid(thickness2, Nx, Ny)
obj.Add_LayerUniform(thickness3, epsilon3)

# 3. Initialize
obj.Init_Setup()

# 4. Input patterns (for grid layers)
obj.GridLayer_geteps(epsilon_grid)

# 5. Define excitation
obj.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order=0)

# 6. Solve
R, T = obj.RT_Solve(normalize=1)

# 7. Analyze fields (optional)
[Ex,Ey,Ez], [Hx,Hy,Hz] = obj.Solve_FieldOnGrid(layer, z, [Nx,Ny])
```

## Coordinate System

GRCWA uses a right-handed Cartesian coordinate system:

- **x, y**: In-plane directions (periodic)
- **z**: Out-of-plane direction (layer stacking)
- Light propagates in the **+z** direction

```
     z ↑
       |
       |  ┌────── Layer N
       |  ├────── Layer 2
       |  ├────── Layer 1
       |  └────── Layer 0 (input)
       └────────→ x
      ↙
     y
```

## Periodicity and Lattice Vectors

### Lattice Definition

Two lattice vectors $\mathbf{L}_1$ and $\mathbf{L}_2$ define the 2D periodic unit cell:

$$
\mathbf{L}_1 = (L_{1x}, L_{1y}, 0)
$$

$$
\mathbf{L}_2 = (L_{2x}, L_{2y}, 0)
$$

The structure repeats every:

$$
\mathbf{r} + m\mathbf{L}_1 + n\mathbf{L}_2 \quad (m, n \in \mathbb{Z})
$$

### Common Lattices

**Square lattice** (period $a$):
```python
L1 = [a, 0]
L2 = [0, a]
```

**Rectangular lattice** (periods $a, b$):
```python
L1 = [a, 0]
L2 = [0, b]
```

**Hexagonal lattice** (period $a$):
```python
L1 = [a, 0]
L2 = [a/2, a*np.sqrt(3)/2]
```

### Unit Cell Area

The area of the unit cell is:

$$
A_{\text{cell}} = |\mathbf{L}_1 \times \mathbf{L}_2| = |L_{1x}L_{2y} - L_{1y}L_{2x}|
$$

## Layer Types

### 1. Uniform Layers

Homogeneous dielectric with constant $\varepsilon$:

```python
obj.Add_LayerUniform(thickness, epsilon)
```

**Advantages:**

- Very fast (analytical solution)
- Numerically stable
- No pattern input needed

**Use for:**

- Vacuum/air regions
- Solid dielectric slabs
- Substrate layers
- Cladding layers

### 2. Grid-Based Patterned Layers

Arbitrary 2D pattern on Cartesian grid:

```python
obj.Add_LayerGrid(thickness, Nx, Ny)
# Later:
obj.GridLayer_geteps(epsilon_grid.flatten())
```

**Advantages:**

- Maximum flexibility (any pattern)
- Easy to define complex shapes
- Supports numerical optimization

**Use for:**

- Photonic crystals
- Metasurfaces
- Arbitrary patterns
- Optimization problems

**Grid resolution:**

- Smooth patterns: Nx, Ny ≈ 50-100
- Sharp features: Nx, Ny ≈ 200-500
- Very fine details: Nx, Ny ≈ 500-1000

### 3. Fourier Series Layers

Pattern defined by analytical Fourier coefficients:

```python
obj.Add_LayerFourier(thickness, params)
```

**Advantages:**

- No FFT needed
- Exact for known geometries (circles, rectangles)

**Disadvantages:**

- Limited to shapes with known Fourier series
- Rarely used in practice

## Truncation Order

### What is it?

The truncation order $N_G$ determines how many Fourier harmonics (diffraction orders) are included:

$$
\mathbf{E}(\mathbf{r}) = \sum_{m,n} \mathbf{E}_{mn}(z) e^{i\mathbf{k}_{mn} \cdot \mathbf{r}_\parallel}
$$

The sum is truncated to $N_G$ terms.

### Choosing $N_G$

**Rules of thumb:**

| Structure Type | Recommended $N_G$ |
|----------------|-------------------|
| Uniform layers | 11-51 |
| Smooth patterns | 51-101 |
| Sharp features | 101-301 |
| Very fine details | 301-501 |
| High accuracy | 501-1001 |

**Trade-offs:**

- ✅ Larger $N_G$ → More accurate
- ❌ Larger $N_G$ → Slower computation
- ❌ Larger $N_G$ → More memory

### Convergence Testing

Always test convergence:

```python
nG_values = [51, 101, 201, 301]
for nG in nG_values:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    # ... setup ...
    R, T = obj.RT_Solve(normalize=1)
    print(f"nG={obj.nG:4d}: R={R:.6f}, T={T:.6f}")
```

Converged when results don't change with increasing $N_G$.

## Incident Wave

### Angles

**Polar angle** $\theta$: Angle from z-axis (surface normal)

- $\theta = 0$: Normal incidence
- $0 < \theta < 90°$: Oblique incidence

**Azimuthal angle** $\phi$: Angle in xy-plane from x-axis

- Defines the direction of the projection of $\mathbf{k}$ onto xy-plane

**Incident wave vector:**

$$
\mathbf{k}_{\text{inc}} = \omega\sqrt{\varepsilon_{\text{in}}}(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)
$$

### Polarization

**P-polarization (TM):**

- Electric field in the plane of incidence
- Magnetic field perpendicular to plane of incidence

**S-polarization (TE):**

- Electric field perpendicular to plane of incidence
- Magnetic field in the plane of incidence

**Arbitrary polarization:**

$$
\mathbf{E} = A_p e^{i\phi_p} \hat{p} + A_s e^{i\phi_s} \hat{s}
$$

**Common cases:**

```python
# P-polarized
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

# S-polarized
obj.MakeExcitationPlanewave(0, 0, 1, 0, 0)

# Linear 45°
obj.MakeExcitationPlanewave(1, 0, 1, 0, 0)

# Left circular
obj.MakeExcitationPlanewave(1, 0, 1, np.pi/2, 0)

# Right circular
obj.MakeExcitationPlanewave(1, 0, 1, -np.pi/2, 0)
```

## Diffraction Orders

### What are they?

Due to periodicity, the incident wave couples to discrete diffraction orders:

$$
\mathbf{k}_{mn,\parallel} = \mathbf{k}_{\parallel,0} + m\mathbf{K}_1 + n\mathbf{K}_2
$$

Each $(m,n)$ is a diffraction order.

### Propagating vs. Evanescent

For each order, compute:

$$
k_{z,mn} = \sqrt{\varepsilon\omega^2 - k_{x,mn}^2 - k_{y,mn}^2}
$$

- **Propagating**: $k_z$ is real → carries power to far field
- **Evanescent**: $k_z$ is imaginary → decays exponentially

**Example:**

```python
# After Init_Setup()
for i in range(obj.nG):
    kx = obj.kx[i]
    ky = obj.ky[i]
    kz_sq = obj.omega**2 - kx**2 - ky**2  # In vacuum
    if kz_sq > 0:
        print(f"Order {obj.G[i]}: propagating, kz={np.sqrt(kz_sq):.4f}")
    else:
        print(f"Order {obj.G[i]}: evanescent, κ={np.sqrt(-kz_sq):.4f}")
```

## Reflection and Transmission

### Total Power

```python
R, T = obj.RT_Solve(normalize=1)
```

- $R$: Total reflected power (all orders)
- $T$: Total transmitted power (all orders)

For lossless structures: $R + T = 1$

### By Order

```python
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
```

- `Ri[i]`: Reflected power in order $i$
- `Ti[i]`: Transmitted power in order $i$

**Analysis:**

```python
print(f"0th order: R={Ri[0]:.4f}, T={Ti[0]:.4f}")
print(f"Higher orders: R={sum(Ri[1:]):.4f}, T={sum(Ti[1:]):.4f}")

# Which orders carry significant power?
threshold = 1e-3
for i in range(obj.nG):
    if Ri[i] > threshold or Ti[i] > threshold:
        m, n = obj.G[i]
        print(f"Order ({m:2d},{n:2d}): R={Ri[i]:.4f}, T={Ti[i]:.4f}")
```

## Field Analysis

### Fourier Space

Get Fourier coefficients:

```python
[Ex_mn, Ey_mn, Ez_mn], [Hx_mn, Hy_mn, Hz_mn] = obj.Solve_FieldFourier(layer, z_offset)
```

Each array has length `nG`, complex values.

### Real Space

Get fields on a grid:

```python
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(layer, z_offset, [Nx, Ny])
```

Each array has shape `(Nx, Ny)`, complex values.

**Intensity:**

```python
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
```

**Poynting vector:**

```python
Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
```

## Normalization

### Why normalize?

Raw RCWA outputs are amplitudes. To get physical powers, we must normalize by:

- Incident power
- Medium properties (impedance)
- Angle (projected area)

### Normalized vs. Unnormalized

```python
# Normalized (recommended)
R, T = obj.RT_Solve(normalize=1)
# R + T = 1 for lossless

# Unnormalized (raw amplitudes)
R_raw, T_raw = obj.RT_Solve(normalize=0)
# Need manual normalization
```

**Always use `normalize=1` for physically meaningful results.**

## Common Pitfalls

### 1. Energy Not Conserved

**Symptom:** $R + T \neq 1$

**Causes:**

- Insufficient $N_G$ (increase truncation order)
- Numerical instability (reduce layer thickness or $N_G$)
- Very high contrast structures (use more grid points)

### 2. Wrong Units

**Symptom:** Strange results, unphysical values

**Solution:** Ensure consistent units:

```python
# All in μm
wavelength = 1.5  # μm
freq = 1.0 / wavelength
L1 = [0.6, 0]  # μm
thickness = 0.3  # μm
```

### 3. Grid Resolution Too Low

**Symptom:** Incorrect patterns, jagged edges

**Solution:** Increase `Nx, Ny`:

```python
# Low resolution (bad)
obj.Add_LayerGrid(0.3, 50, 50)

# High resolution (good)
obj.Add_LayerGrid(0.3, 400, 400)
```

### 4. Pattern Coordinates

**Mistake:** Confusing normalized and physical coordinates

**Correct:**

```python
# Pattern defined on [0,1] × [0,1] (normalized)
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Physical coordinates: multiply by lattice vectors
x_phys = X * L1[0] + Y * L2[0]
y_phys = X * L1[1] + Y * L2[1]
```

### 5. Singular Matrix

**Symptom:** Error in eigenvalue solving

**Causes:**

- Perfect normal incidence with no loss
- Degenerate geometry

**Solution:** Add tiny loss:

```python
# Instead of freq = 1.0
Qabs = 1e8  # Very high Q
freq = 1.0 * (1 + 1j / (2*Qabs))
```

## Best Practices

### ✅ Do

- Test convergence with increasing $N_G$
- Use `normalize=1` for R, T
- Verify $R + T = 1$ for lossless structures
- Use sufficient grid resolution ($N_x, N_y \geq 200$)
- Check energy conservation

### ❌ Don't

- Use too few harmonics (risk unconverged results)
- Forget to call `Init_Setup()` before solving
- Mix units (e.g., μm for length, nm for wavelength)
- Use extremely thick layers (numerical instability)
- Ignore energy conservation errors

## Summary

Key concepts for GRCWA:

1. **Workflow**: Create → Add layers → Initialize → Input patterns → Excite → Solve
2. **Lattice**: Defined by $\mathbf{L}_1, \mathbf{L}_2$
3. **Truncation**: $N_G$ harmonics, test convergence
4. **Layers**: Uniform (fast) or grid-based (flexible)
5. **Excitation**: Angles $(\theta, \phi)$ and polarization $(p, s)$
6. **Results**: $R, T$ total or by order
7. **Fields**: Fourier or real space

## Next Steps

- **[Layer Definition](layers.md)**: Detailed layer management
- **[Excitation Setup](excitation.md)**: Polarization and angles
- **[Computing Results](results.md)**: Interpreting R, T, and fields
- **[Tutorials](../tutorials/tutorial1.md)**: Hands-on examples
