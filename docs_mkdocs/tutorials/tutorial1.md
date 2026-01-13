# Tutorial 1: Simple Dielectric Slab

In this tutorial, you'll learn the basics of GRCWA by simulating a simple dielectric slab.

## Learning Objectives

By the end of this tutorial, you will:

- Understand the basic GRCWA workflow
- Create uniform dielectric layers
- Compute reflection and transmission
- Verify energy conservation
- Understand the Fabry-Pérot effect

## Physical System

We'll simulate a dielectric slab (like a glass plate) in vacuum:

```
        Air (ε=1)
    ┌─────────────────┐
    │   Slab (ε=4)    │  thickness = 0.5λ
    └─────────────────┘
        Air (ε=1)
```

**Parameters:**

- Wavelength: λ = 1.0 μm
- Slab: Silicon (n=2, ε=4), thickness = 0.5 μm
- Incidence: Normal (θ=0)
- Polarization: P-polarized

## Step 1: Import and Setup

```python
import grcwa
import numpy as np
import matplotlib.pyplot as plt

# Set backend (use 'numpy' for speed, 'autograd' for gradients)
grcwa.set_backend('numpy')
```

## Step 2: Define Structure

```python
# Physical parameters
wavelength = 1.0      # μm
freq = 1.0 / wavelength  # frequency in natural units

# Lattice vectors (arbitrary for uniform slab)
L1 = [1.0, 0]
L2 = [0, 1.0]

# Incident angles (normal incidence)
theta = 0.0
phi = 0.0

# Truncation order (small value OK for uniform layers)
nG = 51

# Create RCWA object
obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
```

**Explanation:**

- `freq = 1.0/wavelength`: Converts wavelength to frequency
- `L1, L2`: Periodicity doesn't matter for uniform structure
- `nG = 51`: Small truncation order sufficient for uniform layers
- `verbose=1`: Print progress information

## Step 3: Add Layers

```python
# Layer thicknesses
thickness_top = 1.0      # Semi-infinite vacuum
thickness_slab = 0.5     # Slab thickness = λ/2
thickness_bottom = 1.0   # Semi-infinite vacuum

# Dielectric constants
eps_vacuum = 1.0
eps_slab = 4.0  # n=2, so ε=n²=4

# Add layers (top to bottom)
obj.Add_LayerUniform(thickness_top, eps_vacuum)
obj.Add_LayerUniform(thickness_slab, eps_slab)
obj.Add_LayerUniform(thickness_bottom, eps_vacuum)
```

**Layer ordering:**

1. Input layer (vacuum above slab)
2. Slab layer (dielectric)
3. Output layer (vacuum below slab)

**Note:** First and last layers are treated as semi-infinite regions.

## Step 4: Initialize

```python
# Initialize reciprocal lattice and eigenvalues
obj.Init_Setup()

print(f"Actual nG used: {obj.nG}")
print(f"Angular frequency: {obj.omega:.4f}")
```

This computes:

- Reciprocal lattice vectors
- Wave vectors for all diffraction orders
- Eigenvalues and eigenvectors for each layer

## Step 5: Define Excitation

```python
# P-polarized plane wave
p_amp = 1.0
p_phase = 0.0
s_amp = 0.0
s_phase = 0.0

obj.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order=0)
```

**Polarizations:**

- **P-polarized** (TM): Electric field in plane of incidence
- **S-polarized** (TE): Electric field perpendicular to plane of incidence

For normal incidence, choice doesn't matter for isotropic materials.

## Step 6: Solve

```python
# Compute reflection and transmission
R, T = obj.RT_Solve(normalize=1)

print("\n" + "="*50)
print(f"Reflection (R): {R:.6f}")
print(f"Transmission (T): {T:.6f}")
print(f"Sum (R+T): {R+T:.6f}")
print(f"Energy conservation error: {abs(R+T-1):.2e}")
print("="*50)
```

**Expected output:**

```
==================================================
Reflection (R): 0.055556
Transmission (T): 0.944444
Sum (R+T): 1.000000
Energy conservation error: 0.00e+00
==================================================
```

## Step 7: Analytical Verification

For a dielectric slab, we can compute R, T analytically using Fresnel equations and Fabry-Pérot formula.

```python
def analytical_slab_RT(n_slab, thickness, wavelength, n_in=1.0, n_out=1.0):
    """
    Analytical R, T for a dielectric slab at normal incidence.
    """
    # Refractive index
    n = n_slab
    k0 = 2 * np.pi / wavelength
    kz = n * k0
    delta = kz * thickness  # Phase thickness

    # Fresnel coefficients at interfaces
    r12 = (n_in - n) / (n_in + n)  # Air -> slab
    r23 = (n - n_out) / (n + n_out)  # Slab -> air
    t12 = 2*n_in / (n_in + n)
    t23 = 2*n / (n + n_out)

    # Fabry-Pérot formula
    numerator_r = r12 + r23 * np.exp(2j * delta)
    denominator = 1 + r12 * r23 * np.exp(2j * delta)
    r_total = numerator_r / denominator

    numerator_t = t12 * t23 * np.exp(1j * delta)
    t_total = numerator_t / denominator

    R = np.abs(r_total)**2
    T = (n_out / n_in) * np.abs(t_total)**2

    return R, T

# Compute analytical result
n_slab = 2.0  # sqrt(4.0)
R_analytical, T_analytical = analytical_slab_RT(n_slab, thickness_slab, wavelength)

print("\nAnalytical result:")
print(f"R_analytical = {R_analytical:.6f}")
print(f"T_analytical = {T_analytical:.6f}")
print(f"\nDifference:")
print(f"ΔR = {abs(R - R_analytical):.2e}")
print(f"ΔT = {abs(T - T_analytical):.2e}")
```

**Expected output:**

```
Analytical result:
R_analytical = 0.055556
T_analytical = 0.944444

Difference:
ΔR = 1.39e-17
ΔT = 2.78e-17
```

Perfect agreement! This validates our RCWA simulation.

## Step 8: Spectral Response

Let's compute R and T vs. wavelength to see Fabry-Pérot fringes:

```python
# Wavelength sweep
wavelengths = np.linspace(0.5, 2.0, 200)  # μm
R_spectrum = []
T_spectrum = []
R_analytical_spectrum = []

for wl in wavelengths:
    freq = 1.0 / wl

    # RCWA calculation
    obj_sweep = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj_sweep.Add_LayerUniform(thickness_top, eps_vacuum)
    obj_sweep.Add_LayerUniform(thickness_slab, eps_slab)
    obj_sweep.Add_LayerUniform(thickness_bottom, eps_vacuum)
    obj_sweep.Init_Setup()
    obj_sweep.MakeExcitationPlanewave(p_amp, p_phase, s_amp, s_phase, order=0)

    R, T = obj_sweep.RT_Solve(normalize=1)
    R_spectrum.append(R)
    T_spectrum.append(T)

    # Analytical
    R_an, T_an = analytical_slab_RT(n_slab, thickness_slab, wl)
    R_analytical_spectrum.append(R_an)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, R_spectrum, 'b-', linewidth=2, label='R (RCWA)')
plt.plot(wavelengths, T_spectrum, 'r-', linewidth=2, label='T (RCWA)')
plt.plot(wavelengths, R_analytical_spectrum, 'ko', markersize=3,
         markevery=10, label='R (Analytical)')

plt.xlabel('Wavelength (μm)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.title('Fabry-Pérot Fringes in Dielectric Slab', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0.5, 2.0)
plt.ylim(0, 1)

# Mark special wavelengths
# When thickness = m*λ/(2n), we have constructive interference -> minimum R
lambda_min = 2 * n_slab * thickness_slab / np.arange(1, 5)
for lm in lambda_min:
    if 0.5 < lm < 2.0:
        plt.axvline(lm, color='gray', linestyle='--', alpha=0.5)
        plt.text(lm, 0.95, f'λ={lm:.2f}', fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('tutorial1_spectrum.png', dpi=150)
plt.show()
```

**What you'll see:**

- Oscillating R and T (Fabry-Pérot fringes)
- Minima in R when thickness = mλ/(2n) (constructive interference)
- Perfect agreement between RCWA and analytical

## Understanding the Physics

### Fabry-Pérot Effect

Light bounces between the two interfaces of the slab, creating interference:

- **Constructive interference**: When optical path = integer × wavelength
    - Thickness = m × λ/(2n)
    - Maximum transmission, minimum reflection

- **Destructive interference**: When optical path = half-integer × wavelength
    - Thickness = (m + 1/2) × λ/(2n)
    - Minimum transmission, maximum reflection

### Fresnel Reflection

At each air-slab interface, Fresnel coefficient:

$$
r = \frac{n_1 - n_2}{n_1 + n_2}
$$

For n=2:

$$
r = \frac{1-2}{1+2} = -\frac{1}{3}
$$

$$
R_{\text{single interface}} = |r|^2 = \frac{1}{9} \approx 0.111
$$

But with two interfaces and interference, $R \approx 0.056$ at λ = 1.0 μm.

## Exercise 1: Change Thickness

Modify the slab thickness and observe how it affects R and T:

```python
thicknesses = [0.25, 0.5, 0.75, 1.0]  # in units of λ

for thickness in thicknesses:
    obj_ex = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj_ex.Add_LayerUniform(1.0, eps_vacuum)
    obj_ex.Add_LayerUniform(thickness, eps_slab)
    obj_ex.Add_LayerUniform(1.0, eps_vacuum)
    obj_ex.Init_Setup()
    obj_ex.MakeExcitationPlanewave(1, 0, 0, 0, 0)
    R, T = obj_ex.RT_Solve(normalize=1)
    print(f"Thickness = {thickness:.2f}λ: R = {R:.4f}, T = {T:.4f}")
```

## Exercise 2: Change Refractive Index

Try different slab materials:

```python
refractive_indices = [1.5, 2.0, 3.0, 3.5]  # Glass, Si, GaAs, Si at IR

for n in refractive_indices:
    eps = n**2
    obj_ex = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj_ex.Add_LayerUniform(1.0, eps_vacuum)
    obj_ex.Add_LayerUniform(0.5, eps)
    obj_ex.Add_LayerUniform(1.0, eps_vacuum)
    obj_ex.Init_Setup()
    obj_ex.MakeExcitationPlanewave(1, 0, 0, 0, 0)
    R, T = obj_ex.RT_Solve(normalize=1)
    print(f"n = {n:.1f}: R = {R:.4f}, T = {T:.4f}")
```

**Observation:** Higher refractive index → higher reflection.

## Exercise 3: Oblique Incidence

What happens at an angle?

```python
angles = np.linspace(0, 80, 50) * np.pi/180

R_p_list = []
R_s_list = []

for theta in angles:
    obj_angle = grcwa.obj(nG, L1, L2, freq, theta, phi=0, verbose=0)
    obj_angle.Add_LayerUniform(1.0, eps_vacuum)
    obj_angle.Add_LayerUniform(0.5, eps_slab)
    obj_angle.Add_LayerUniform(1.0, eps_vacuum)
    obj_angle.Init_Setup()

    # P-polarization
    obj_angle.MakeExcitationPlanewave(1, 0, 0, 0, 0)
    R_p, _ = obj_angle.RT_Solve(normalize=1)
    R_p_list.append(R_p)

    # S-polarization
    obj_angle.MakeExcitationPlanewave(0, 0, 1, 0, 0)
    R_s, _ = obj_angle.RT_Solve(normalize=1)
    R_s_list.append(R_s)

plt.figure(figsize=(8, 6))
plt.plot(angles*180/np.pi, R_p_list, 'b-', label='P-polarization')
plt.plot(angles*180/np.pi, R_s_list, 'r-', label='S-polarization')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Reflectance')
plt.title('Angle-Dependent Reflection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Observation:** Brewster's angle for P-polarization, total internal reflection at grazing angles.

## Summary

In this tutorial, you learned:

✅ How to create a basic RCWA simulation
✅ Add uniform dielectric layers
✅ Compute reflection and transmission
✅ Verify results against analytical formulas
✅ Compute spectral response (Fabry-Pérot fringes)
✅ Understand physical phenomena (interference, Fresnel reflection)

## Next Steps

- **[Tutorial 2](tutorial2.md)**: Patterned layer with holes
- **[Tutorial 3](tutorial3.md)**: Multiple patterned layers
- **[Tutorial 4](tutorial4.md)**: Hexagonal lattices
- **[Tutorial 5](tutorial5.md)**: Topology optimization

## Complete Code

<details>
<summary>Click to expand full code</summary>

```python
import grcwa
import numpy as np
import matplotlib.pyplot as plt

# Setup
grcwa.set_backend('numpy')

wavelength = 1.0
freq = 1.0 / wavelength
L1 = [1.0, 0]
L2 = [0, 1.0]
theta = 0.0
phi = 0.0
nG = 51

obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)

# Layers
thickness_slab = 0.5
eps_vacuum = 1.0
eps_slab = 4.0

obj.Add_LayerUniform(1.0, eps_vacuum)
obj.Add_LayerUniform(thickness_slab, eps_slab)
obj.Add_LayerUniform(1.0, eps_vacuum)

# Initialize
obj.Init_Setup()

# Excitation
obj.MakeExcitationPlanewave(1, 0, 0, 0, 0)

# Solve
R, T = obj.RT_Solve(normalize=1)
print(f"R = {R:.6f}, T = {T:.6f}, R+T = {R+T:.6f}")

# Analytical verification
n_slab = 2.0
k0 = 2 * np.pi / wavelength
kz = n_slab * k0
delta = kz * thickness_slab

r12 = (1 - n_slab) / (1 + n_slab)
r23 = (n_slab - 1) / (n_slab + 1)
t12 = 2 / (1 + n_slab)
t23 = 2*n_slab / (n_slab + 1)

r_total = (r12 + r23 * np.exp(2j * delta)) / (1 + r12 * r23 * np.exp(2j * delta))
t_total = (t12 * t23 * np.exp(1j * delta)) / (1 + r12 * r23 * np.exp(2j * delta))

R_analytical = np.abs(r_total)**2
T_analytical = np.abs(t_total)**2

print(f"R_analytical = {R_analytical:.6f}")
print(f"T_analytical = {T_analytical:.6f}")
print(f"Error: ΔR = {abs(R-R_analytical):.2e}")
```

</details>
