# Example Gallery

Browse examples of RCWA simulations with GRCWA.

## Basic Examples

### Example 1: Square Lattice with Circular Holes

Compute transmission and reflection through a photonic crystal slab with circular air holes in a square lattice.

**Structure:**

- Square lattice: period = 1.5 μm
- Silicon slab (ε=4) with circular air holes
- Hole radius: 0.3 × period
- Slab thickness: 0.2 μm

**Results:**

- Total and per-order reflection/transmission
- Demonstrates diffraction into multiple orders

---

### Example 2: Two Patterned Layers

Multi-layer structure with two different patterned layers.

**Structure:**

- Layer 1: Circular holes (ε=4)
- Layer 2: Square holes (ε=6)
- Different patterns in each layer

**Results:**

- Shows how to handle multiple patterned layers
- Oblique incidence (θ = π/10)

---

### Example 3: Topology Optimization

Inverse design using automatic differentiation.

**Objective:** Maximize reflection from a single patterned layer

**Method:**

- Autograd for gradients
- NLOPT for optimization
- Gradient-based topology optimization

**Code:** [ex3.py](ex3.md)

**Results:**

- Optimized dielectric pattern
- Convergence of reflection vs. iteration

---

### Example 4: Hexagonal Lattice

Hexagonal lattice of circular holes.

**Structure:**

- Hexagonal lattice (60° angle)
- Circular air holes
- High grid resolution (1000×1000)

**Code:** [ex4.py](ex4.md)

**Results:**

- Demonstrates non-orthogonal lattices
- Proper coordinate transformation for hexagonal symmetry

---

## Advanced Examples

### Bragg Mirror

Multi-layer Bragg reflector (1D photonic crystal).

```python
import grcwa
import numpy as np

# Parameters
wavelength = 1.0
freq = 1.0 / wavelength
n1, n2 = 2.0, 1.5  # High/low index
d1 = wavelength / (4*n1)  # Quarter-wave thickness
d2 = wavelength / (4*n2)
N_pairs = 10  # Number of layer pairs

# Setup
obj = grcwa.obj(51, [1,0], [0,1], freq, 0, 0, verbose=0)

# Add layers
for i in range(N_pairs):
    obj.Add_LayerUniform(d1, n1**2)
    obj.Add_LayerUniform(d2, n2**2)

obj.Init_Setup()
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
R, T = obj.RT_Solve(normalize=1)

print(f"Bragg Mirror: R={R:.4f}, T={T:.4f}")
```

**Result:** High reflection (~99%) at design wavelength.

---

### Anti-Reflection Coating

Quarter-wave anti-reflection coating.

```python
# Substrate: n=3.5 (e.g., GaAs)
# Coating: n=sqrt(3.5) ≈ 1.87 (optimal)
# Thickness: λ/4n

wavelength = 1.0
n_substrate = 3.5
n_coating = np.sqrt(n_substrate)
thickness = wavelength / (4*n_coating)

obj = grcwa.obj(51, [1,0], [0,1], 1/wavelength, 0, 0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)  # Air
obj.Add_LayerUniform(thickness, n_coating**2)  # AR coating
obj.Add_LayerUniform(10.0, n_substrate**2)  # Substrate (thick)

obj.Init_Setup()
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
R, T = obj.RT_Solve(normalize=1)

print(f"Without AR coating, R would be: {((1-n_substrate)/(1+n_substrate))**2:.4f}")
print(f"With AR coating: R={R:.6f}")
```

**Result:** Nearly zero reflection at design wavelength.

---

### Metasurface Lens

Phase gradient metasurface for beam steering.

```python
# Create phase gradient across unit cell
def phase_gradient_pattern(Nx, Ny, gradient_angle):
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Phase = k*x*sin(theta) approximated by varying dielectric
    # Simplified model: use pillars of varying size
    phase = 2*np.pi*X*np.tan(gradient_angle)
    pillar_radius = 0.1 + 0.2 * (phase / (2*np.pi)) % 1

    eps = np.ones((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if (X[i,j]-0.5)**2 + (Y[i,j]-0.5)**2 < pillar_radius**2:
                eps[i,j] = 12.0  # Silicon

    return eps

# Simulate
gradient_angle = 10 * np.pi/180
eps_meta = phase_gradient_pattern(300, 300, gradient_angle)

obj = grcwa.obj(201, [0.6,0], [0,0.6], 1.0, 0, 0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)
obj.Add_LayerGrid(0.5, 300, 300)
obj.Add_LayerUniform(1.0, 1.0)
obj.Init_Setup()
obj.GridLayer_geteps(eps_meta.flatten())
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

# Find dominant diffraction order
max_order = np.argmax(Ti)
print(f"Most power in order {obj.G[max_order]}: T={Ti[max_order]:.4f}")
```

---

### Photonic Crystal Waveguide

Line defect in photonic crystal.

```python
# Create photonic crystal with line defect
def pc_waveguide(Nx, Ny, lattice_const=0.4, hole_radius=0.3):
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    eps = np.ones((Nx, Ny)) * 12.0  # Silicon background

    # Regular array of holes
    for i in range(-2, 3):
        for j in range(-2, 3):
            if j == 0:
                continue  # Skip middle row (waveguide)
            cx = 0.5 + i*lattice_const
            cy = 0.5 + j*lattice_const
            hole = (X-cx)**2 + (Y-cy)**2 < hole_radius**2
            eps[hole] = 1.0

    return eps

eps_wg = pc_waveguide(400, 400)

# Simulate guided mode at frequency in band gap
obj = grcwa.obj(201, [2.0,0], [0,2.0], 0.3, 0, 0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)
obj.Add_LayerGrid(1.0, 400, 400)
obj.Add_LayerUniform(1.0, 1.0)
obj.Init_Setup()
obj.GridLayer_geteps(eps_wg.flatten())
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

R, T = obj.RT_Solve(normalize=1)
print(f"Guided mode: R={R:.4f}, T={T:.4f}")
```

---

### Grating Coupler

Couples normally incident light to in-plane waveguide mode.

```python
# 1D grating
def grating(Nx, Ny, period_frac=0.5, depth_frac=0.3):
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Background: Silicon
    eps = np.ones((Nx, Ny)) * 12.0

    # Grating: remove silicon in alternating stripes
    grating_lines = (X % (1/5)) < (period_frac / 5)
    eps[grating_lines] = 1.0

    return eps

eps_grating = grating(400, 200)

obj = grcwa.obj(201, [0.6,0], [0,0.3], 1.0, 0, 0, verbose=0)
obj.Add_LayerUniform(1.0, 1.0)
obj.Add_LayerGrid(0.2, 400, 200)
obj.Add_LayerUniform(1.0, 12.0)  # Silicon substrate
obj.Init_Setup()
obj.GridLayer_geteps(eps_grating.flatten())
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

# Power coupled to different orders
for i in range(min(10, obj.nG)):
    if Ti[i] > 0.01:
        print(f"Order {obj.G[i]}: T={Ti[i]:.4f}")
```

---

## Spectral Calculations

### Photonic Band Diagram

Compute transmission vs. frequency and incident angle.

```python
frequencies = np.linspace(0.3, 0.7, 40)
angles = np.linspace(0, 45, 30) * np.pi/180

T_map = np.zeros((len(frequencies), len(angles)))

for i, freq in enumerate(frequencies):
    for j, theta in enumerate(angles):
        obj = grcwa.obj(101, [1,0], [0,1], freq, theta, 0, verbose=0)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerGrid(0.5, 200, 200)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Init_Setup()
        obj.GridLayer_geteps(eps_pattern.flatten())
        obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)
        R, T = obj.RT_Solve(normalize=1)
        T_map[i,j] = T

# Plot band diagram
plt.figure(figsize=(8,6))
plt.pcolormesh(angles*180/np.pi, frequencies, T_map, cmap='hot', shading='auto')
plt.colorbar(label='Transmission')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Frequency (c/λ)')
plt.title('Photonic Band Diagram')
plt.show()
```

---

## Visualization Examples

### Field Intensity Profile

```python
# After solving
[Ex, Ey, Ez], [Hx, Hy, Hz] = obj.Solve_FieldOnGrid(1, 0.15, [200, 200])

# Intensity
I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

# Plot
plt.figure(figsize=(8,8))
plt.imshow(I.T, origin='lower', cmap='hot', extent=[0, L1[0], 0, L2[1]])
plt.colorbar(label='|E|²')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.title('Electric Field Intensity')
plt.tight_layout()
plt.show()
```

### Poynting Vector Field

```python
# Compute Poynting vector
Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))

# Plot
plt.figure(figsize=(8,8))
plt.quiver(Sx[::10,::10], Sy[::10,::10])
plt.imshow(np.abs(Sz.T), origin='lower', cmap='coolwarm', alpha=0.5)
plt.colorbar(label='Sz')
plt.title('Poynting Vector (energy flow)')
plt.show()
```

---

## Tips for Creating Your Own Examples

1. **Start simple**: Begin with uniform layers to verify setup
2. **Test convergence**: Increase $N_G$ until results stabilize
3. **Check energy conservation**: $R + T = 1$ for lossless
4. **Visualize patterns**: Use `Return_eps()` to verify your pattern
5. **Analyze fields**: Use `Solve_FieldOnGrid()` to understand physics
6. **Sweep parameters**: Vary wavelength, angle, or geometry

## See Also

- **[Tutorials](../tutorials/tutorial1.md)**: Step-by-step guided examples
- **[API Reference](../api/core.md)**: Complete function documentation
- **[FAQ](../reference/faq.md)**: Frequently asked questions
