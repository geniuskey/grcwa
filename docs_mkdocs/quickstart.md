# Quick Start Guide

This guide will get you running your first RCWA simulation in minutes!

## Your First Simulation

Let's simulate a simple dielectric slab (like a glass plate) and compute how much light it reflects and transmits.

### Complete Example

```python
import grcwa
import numpy as np

# Step 1: Define the structure parameters
L1 = [1.0, 0]    # Lattice vector 1 (x-direction)
L2 = [0, 1.0]    # Lattice vector 2 (y-direction)
freq = 1.0       # Frequency (wavelength = 1.0 in our units)
theta = 0.0      # Normal incidence
phi = 0.0        # Azimuthal angle
nG = 101         # Number of Fourier harmonics

# Step 2: Create RCWA object
obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)

# Step 3: Define layers (from top to bottom)
obj.Add_LayerUniform(1.0, 1.0)   # Top: vacuum (ε=1), thickness=1.0
obj.Add_LayerUniform(0.5, 4.0)   # Middle: dielectric slab (ε=4), thickness=0.5
obj.Add_LayerUniform(1.0, 1.0)   # Bottom: vacuum (ε=1), thickness=1.0

# Step 4: Initialize
obj.Init_Setup()

# Step 5: Define incident wave (p-polarized, amplitude=1)
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0,
                            order=0)

# Step 6: Solve
R, T = obj.RT_Solve(normalize=1)

# Step 7: Display results
print(f"Reflection: R = {R:.4f}")
print(f"Transmission: T = {T:.4f}")
print(f"Energy conservation: R + T = {R+T:.4f}")
```

**Output:**
```
Reflection: R = 0.3600
Transmission: T = 0.6400
Energy conservation: R + T = 1.0000
```

## Understanding the Code

### Step 1: Structure Parameters

```python
L1 = [1.0, 0]    # Lattice vector 1
L2 = [0, 1.0]    # Lattice vector 2
```

These define the periodicity of your structure. For this simple slab, periodicity doesn't matter, but RCWA always assumes periodic structures.

```python
freq = 1.0
```

In natural units (c=1), frequency = 1/wavelength. So `freq=1.0` means wavelength=1.0.

```python
theta = 0.0  # Angle from z-axis (normal incidence)
phi = 0.0    # Angle in xy-plane
```

For normal incidence, set both angles to zero.

```python
nG = 101
```

Number of Fourier harmonics to include. Higher = more accurate but slower. For uniform slabs, even `nG=1` would work, but for patterned layers you need larger values (100-300 typical).

### Step 2: Create RCWA Object

```python
obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
```

This creates the main RCWA solver object. `verbose=1` prints progress information.

### Step 3: Add Layers

```python
obj.Add_LayerUniform(thickness, epsilon)
```

Layers are added from **top to bottom** (input to output):

- **Layer 0**: Top vacuum (semi-infinite)
- **Layer 1**: Dielectric slab
- **Layer 2**: Bottom vacuum (semi-infinite)

!!! note "Layer Order"
    Always define layers in the order light encounters them:
    Input region → Layer 1 → Layer 2 → ... → Output region

### Step 4: Initialize

```python
obj.Init_Setup()
```

This computes:
- Reciprocal lattice vectors
- Wave vectors for all diffraction orders
- Eigenvalues for uniform layers

### Step 5: Define Excitation

```python
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0,
                            order=0)
```

Parameters:
- `p_amp`: P-polarization amplitude (TM)
- `s_amp`: S-polarization amplitude (TE)
- `p_phase`, `s_phase`: Phases in radians
- `order=0`: Incident in the 0th diffraction order (normally incident)

### Step 6: Solve

```python
R, T = obj.RT_Solve(normalize=1)
```

- `normalize=1`: Normalize by incident power and medium properties
- Returns total reflection (R) and transmission (T) powers

### Step 7: Check Energy Conservation

For lossless materials:
```python
assert abs(R + T - 1.0) < 1e-6, "Energy not conserved!"
```

## Example 2: Patterned Layer

Now let's simulate something more interesting: a photonic crystal slab with circular holes.

```python
import grcwa
import numpy as np

# Setup
L1 = [1.5, 0]
L2 = [0, 1.5]
freq = 1.0
theta = 0.0
phi = 0.0
nG = 201  # Need more harmonics for patterned layers

obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)

# Layer structure
obj.Add_LayerUniform(1.0, 1.0)        # vacuum
obj.Add_LayerGrid(0.3, 400, 400)      # patterned layer: 400×400 grid
obj.Add_LayerUniform(1.0, 1.0)        # vacuum

obj.Init_Setup()

# Create pattern: Silicon slab with circular air hole
Nx, Ny = 400, 400
x = np.linspace(0, 1, Nx)  # Normalized coordinates [0,1]
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Start with silicon (ε=12)
epsilon = np.ones((Nx, Ny)) * 12.0

# Add circular air hole (ε=1) at center
radius = 0.4  # In units of lattice constant
hole = (X - 0.5)**2 + (Y - 0.5)**2 < radius**2
epsilon[hole] = 1.0

# Input the pattern
obj.GridLayer_geteps(epsilon.flatten())

# Excitation
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                            s_amp=0, s_phase=0, order=0)

# Solve
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")

# Get reflection/transmission by order
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
print(f"\nNumber of diffraction orders: {len(Ri)}")
print(f"Reflection in 0th order: {Ri[0]:.4f}")
print(f"Transmission in 0th order: {Ti[0]:.4f}")
```

### Key Differences

**Grid-based layer:**
```python
obj.Add_LayerGrid(thickness, Nx, Ny)
```

Instead of uniform dielectric, we define a 2D grid with `Nx × Ny` points.

**Pattern definition:**
```python
epsilon = np.ones((Nx, Ny)) * 12.0  # Background
epsilon[hole] = 1.0                  # Hole
obj.GridLayer_geteps(epsilon.flatten())
```

Create a 2D array of dielectric constants, then flatten and input.

**By-order analysis:**
```python
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
```

Get arrays of R and T for each diffraction order.

## Example 3: Angle-Dependent Response

Compute reflection vs. incident angle:

```python
import grcwa
import numpy as np
import matplotlib.pyplot as plt

# Setup structure
L1 = [0.6, 0]
L2 = [0, 0.6]
freq = 1.0
nG = 101

# Angle sweep
angles = np.linspace(0, 80, 50) * np.pi/180  # 0° to 80°
R_list = []

for theta in angles:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi=0, verbose=0)

    # Bragg mirror: alternating layers
    for i in range(5):
        if i % 2 == 0:
            obj.Add_LayerUniform(0.125, 4.0)  # High index
        else:
            obj.Add_LayerUniform(0.125, 2.25) # Low index

    obj.Init_Setup()
    obj.MakeExcitationPlanewave(1, 0, 0, 0, order=0)

    R, T = obj.RT_Solve(normalize=1)
    R_list.append(R)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(angles * 180/np.pi, R_list, 'b-', linewidth=2)
plt.xlabel('Incident Angle (degrees)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.title('Angle-Dependent Reflection of Bragg Mirror')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('angle_sweep.png', dpi=150)
plt.show()
```

## Common Workflows

### Spectral Calculation

```python
wavelengths = np.linspace(0.4, 0.8, 100)  # μm
freqs = 1.0 / wavelengths

R_spectrum = []
T_spectrum = []

for freq in freqs:
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    # ... add layers ...
    obj.Init_Setup()
    # ... setup excitation and pattern ...
    R, T = obj.RT_Solve(normalize=1)
    R_spectrum.append(R)
    T_spectrum.append(T)

# Plot spectrum
plt.plot(wavelengths, R_spectrum, label='R')
plt.plot(wavelengths, T_spectrum, label='T')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Power')
plt.legend()
```

### Field Visualization

```python
# After solving for R, T
layer = 1  # Which layer to visualize
z_offset = 0.5  # Position within layer
Nxy = [100, 100]  # Grid resolution

# Get fields on grid
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(layer, z_offset, Nxy)

# Compute intensity
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(I.T, origin='lower', cmap='hot', extent=[0, L1[0], 0, L2[1]])
plt.colorbar(label='Intensity |E|²')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Field Intensity in Layer {layer}')
plt.show()
```

## Quick Reference

### Layer Types

| Method | Use Case |
|--------|----------|
| `Add_LayerUniform(thickness, ε)` | Homogeneous dielectric |
| `Add_LayerGrid(thickness, Nx, Ny)` | Arbitrary 2D pattern |
| `Add_LayerFourier(thickness, params)` | Analytical Fourier series |

### Solving Options

```python
# Total R, T
R, T = obj.RT_Solve(normalize=1)

# By diffraction order
Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

# Fields in Fourier space
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldFourier(layer, z_offset)

# Fields in real space
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(layer, z_offset, [Nx, Ny])
```

### Polarization

**P-polarization (TM)**: Electric field in plane of incidence
```python
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=0, s_phase=0, order=0)
```

**S-polarization (TE)**: Electric field perpendicular to plane of incidence
```python
obj.MakeExcitationPlanewave(p_amp=0, p_phase=0, s_amp=1, s_phase=0, order=0)
```

**Circular polarization**:
```python
# Left circular
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=1, s_phase=np.pi/2, order=0)

# Right circular
obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=1, s_phase=-np.pi/2, order=0)
```

## Tips for Beginners

1. **Start simple**: Begin with uniform layers before trying patterned structures
2. **Check energy conservation**: `R + T` should equal 1.0 for lossless materials
3. **Use enough harmonics**: For patterned layers, `nG=101-301` typical
4. **Grid resolution**: Use `Nx, Ny ≥ 200` for accurate patterns
5. **Avoid singularities**: For perfect normal incidence with no absorption, add tiny loss
6. **Normalize**: Always use `normalize=1` for physically meaningful R, T

## Next Steps

You're now ready to explore more advanced features:

- **[Basic Concepts](guide/concepts.md)**: Understand RCWA theory in depth
- **[Tutorials](tutorials/tutorial1.md)**: Step-by-step guided examples
- **[Examples](examples/gallery.md)**: Browse the example gallery
- **[API Reference](api/core.md)**: Detailed function documentation

## Need Help?

- Check the [FAQ](reference/faq.md)
- Read [Troubleshooting](reference/troubleshooting.md)
- See [full examples](examples/gallery.md)
