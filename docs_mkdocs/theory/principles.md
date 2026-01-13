# Physical Principles of RCWA

This page explains the physics behind Rigorous Coupled Wave Analysis (RCWA).

## Maxwell's Equations

RCWA solves Maxwell's equations for time-harmonic fields with convention $e^{-i\omega t}$:

### Curl Equations

$$
\nabla \times \mathbf{E} = i\omega \mathbf{B} = i\omega \mu \mathbf{H}
$$

$$
\nabla \times \mathbf{H} = -i\omega \mathbf{D} = -i\omega \varepsilon \mathbf{E}
$$

### Divergence Equations

$$
\nabla \cdot \mathbf{D} = \nabla \cdot (\varepsilon \mathbf{E}) = 0
$$

$$
\nabla \cdot \mathbf{B} = \nabla \cdot (\mu \mathbf{H}) = 0
$$

In GRCWA's natural units: $\varepsilon_0 = \mu_0 = c = 1$

## Wave Equation

From Maxwell's curl equations, we can derive the vector wave equation:

$$
\nabla \times \nabla \times \mathbf{E} - \omega^2 \varepsilon \mathbf{E} = 0
$$

$$
\nabla \times \nabla \times \mathbf{H} - \omega^2 \mu \mathbf{H} = 0
$$

Using the identity $\nabla \times \nabla \times \mathbf{F} = \nabla(\nabla \cdot \mathbf{F}) - \nabla^2 \mathbf{F}$:

For **E-field** in uniform media:
$$
\nabla^2 \mathbf{E} + \omega^2 \varepsilon \mathbf{E} = 0
$$

This is the Helmholtz equation with wave number $k = \omega\sqrt{\varepsilon}$.

## Periodic Structures

### Lattice Periodicity

A photonic crystal has 2D periodicity:

$$
\varepsilon(\mathbf{r} + m\mathbf{L}_1 + n\mathbf{L}_2) = \varepsilon(\mathbf{r})
$$

where $m, n \in \mathbb{Z}$ and $\mathbf{L}_1, \mathbf{L}_2$ are lattice vectors.

**Example lattices:**

- **Square**: $\mathbf{L}_1 = a\hat{x}$, $\mathbf{L}_2 = a\hat{y}$
- **Rectangular**: $\mathbf{L}_1 = a\hat{x}$, $\mathbf{L}_2 = b\hat{y}$
- **Hexagonal**: $\mathbf{L}_1 = a\hat{x}$, $\mathbf{L}_2 = a(\frac{1}{2}\hat{x} + \frac{\sqrt{3}}{2}\hat{y})$

### Reciprocal Lattice

The reciprocal lattice vectors satisfy:

$$
\mathbf{K}_i \cdot \mathbf{L}_j = 2\pi \delta_{ij}
$$

For 2D lattices embedded in 3D:

$$
\mathbf{K}_1 = 2\pi \frac{\mathbf{L}_2 \times \hat{z}}{|\mathbf{L}_1 \times \mathbf{L}_2 \cdot \hat{z}|}
$$

$$
\mathbf{K}_2 = 2\pi \frac{\hat{z} \times \mathbf{L}_1}{|\mathbf{L}_1 \times \mathbf{L}_2 \cdot \hat{z}|}
$$

The reciprocal lattice vectors are:

$$
\mathbf{G}_{mn} = m\mathbf{K}_1 + n\mathbf{K}_2, \quad m,n \in \mathbb{Z}
$$

## Bloch's Theorem

### Bloch Waves

In periodic structures, electromagnetic modes are **Bloch waves**:

$$
\mathbf{E}(\mathbf{r}) = e^{i\mathbf{k}_\parallel \cdot \mathbf{r}_\parallel} \mathbf{u}(\mathbf{r})
$$

where:

- $\mathbf{k}_\parallel = k_x\hat{x} + k_y\hat{y}$ is the Bloch wave vector
- $\mathbf{u}(\mathbf{r})$ has the same periodicity as the structure: $\mathbf{u}(\mathbf{r} + \mathbf{L}) = \mathbf{u}(\mathbf{r})$

### Floquet-Bloch Theorem

Since $\mathbf{u}(\mathbf{r})$ is periodic, it can be expanded in a Fourier series:

$$
\mathbf{E}(\mathbf{r}) = e^{i\mathbf{k}_\parallel \cdot \mathbf{r}_\parallel} \sum_{mn} \mathbf{E}_{mn}(z) e^{i\mathbf{G}_{mn} \cdot \mathbf{r}_\parallel}
$$

$$
= \sum_{mn} \mathbf{E}_{mn}(z) e^{i(\mathbf{k}_\parallel + \mathbf{G}_{mn}) \cdot \mathbf{r}_\parallel}
$$

Define **Bloch wave vectors**:

$$
\mathbf{k}_{mn,\parallel} = \mathbf{k}_\parallel + \mathbf{G}_{mn}
$$

Each $(m,n)$ is a **diffraction order** or **Floquet harmonic**.

## Diffraction and Propagation

### In-Plane Wave Vectors

For incident plane wave with angles $(\theta, \phi)$ in a medium with $\varepsilon_{\text{in}}$:

$$
k_{x0} = \omega\sqrt{\varepsilon_{\text{in}}} \sin\theta \cos\phi
$$

$$
k_{y0} = \omega\sqrt{\varepsilon_{\text{in}}} \sin\theta \sin\phi
$$

Each diffraction order has in-plane wave vector:

$$
k_{x,mn} = k_{x0} + G_{x,mn}
$$

$$
k_{y,mn} = k_{y0} + G_{y,mn}
$$

### Z-Component of Wave Vector

From the dispersion relation in a medium with $\varepsilon$:

$$
k_{x,mn}^2 + k_{y,mn}^2 + k_{z,mn}^2 = \varepsilon \omega^2
$$

Therefore:

$$
k_{z,mn} = \pm\sqrt{\varepsilon \omega^2 - k_{x,mn}^2 - k_{y,mn}^2}
$$

**Two cases:**

1. **Propagating modes**: $k_{x,mn}^2 + k_{y,mn}^2 < \varepsilon\omega^2$ → $k_z$ is real
   - Carries power to far field
   - Contributes to reflection/transmission

2. **Evanescent modes**: $k_{x,mn}^2 + k_{y,mn}^2 > \varepsilon\omega^2$ → $k_z$ is imaginary
   - Decays exponentially: $e^{-|\text{Im}(k_z)|z}$
   - Stores near-field energy
   - Important for field distributions but doesn't contribute to R/T

### Physical Interpretation

Think of diffraction orders as different "beams" produced by the periodic structure:

- **0th order** $(m=0, n=0)$: Main transmitted/reflected beam
- **Higher orders** $(m,n \neq 0)$: Diffracted beams at angles determined by $\mathbf{k}_{mn}$

At low frequencies (long wavelengths), most orders are evanescent. At high frequencies, many orders can propagate.

## Polarization

### S and P Polarization

For plane waves, we decompose into two orthogonal polarizations:

**S-polarization (TE)**: Electric field perpendicular to plane of incidence

- For normal incidence with $\mathbf{k} \parallel \hat{z}$: $\mathbf{E} \parallel \hat{x}$ or $\hat{y}$

**P-polarization (TM)**: Magnetic field perpendicular to plane of incidence

- Electric field in the plane of incidence

### General Polarization State

An arbitrary polarization is a superposition:

$$
\mathbf{E} = A_s e^{i\phi_s} \hat{s} + A_p e^{i\phi_p} \hat{p}
$$

**Linear polarization**: $\phi_p - \phi_s = 0$ or $\pi$

**Circular polarization**: $A_p = A_s$, $\phi_p - \phi_s = \pm\pi/2$

**Elliptical polarization**: General case

## Energy and Power Flow

### Poynting Vector

The time-averaged Poynting vector is:

$$
\langle\mathbf{S}\rangle = \frac{1}{2} \text{Re}(\mathbf{E} \times \mathbf{H}^*)
$$

This gives the electromagnetic power flow (energy per area per time).

### Power in Each Order

For each diffraction order $(m,n)$:

$$
P_{mn} = \frac{1}{2} \text{Re}\left( E_{x,mn} H_{y,mn}^* - E_{y,mn} H_{x,mn}^* \right) \cdot \frac{A_{\text{cell}}}{\cos\theta_{mn}}
$$

where:

- $A_{\text{cell}} = |\mathbf{L}_1 \times \mathbf{L}_2|$ is the unit cell area
- $\theta_{mn}$ is the propagation angle of order $(m,n)$

### Reflection and Transmission

**Total reflected power**:

$$
R = \sum_{mn} \frac{P_{mn}^{\text{refl}}}{P_{\text{inc}}}
$$

**Total transmitted power**:

$$
T = \sum_{mn} \frac{P_{mn}^{\text{trans}}}{P_{\text{inc}}}
$$

For lossless structures: $R + T = 1$ (energy conservation)

## Boundary Conditions

### Tangential Field Continuity

At interfaces between layers, tangential components must be continuous:

$$
\mathbf{n} \times (\mathbf{E}_1 - \mathbf{E}_2) = 0
$$

$$
\mathbf{n} \times (\mathbf{H}_1 - \mathbf{H}_2) = 0
$$

where $\mathbf{n}$ is the interface normal.

For horizontal interfaces ($\mathbf{n} = \hat{z}$):

$$
E_x, E_y, H_x, H_y \text{ continuous}
$$

### Normal Field Discontinuity

Normal components satisfy:

$$
\mathbf{n} \cdot (\varepsilon_1 \mathbf{E}_1 - \varepsilon_2 \mathbf{E}_2) = 0
$$

$$
\mathbf{n} \cdot (\mu_1 \mathbf{H}_1 - \mu_2 \mathbf{H}_2) = 0
$$

For dielectric contrast, $D_z = \varepsilon E_z$ is continuous, but $E_z$ is not.

## Layer Structure in RCWA

### Uniform Layers

For layers with constant $\varepsilon(z) = \varepsilon_0$:

- Analytical eigenmode solution
- Modes are forward/backward propagating plane waves
- Fast and stable

### Patterned Layers

For layers with $\varepsilon(x,y,z) = \varepsilon(x,y)$ (z-invariant):

- Must solve eigenvalue problem numerically
- Modes are Bloch waves with different $k_z$
- More computationally expensive

### Layer Stacking

RCWA treats the structure as a stack:

```
Input region (uniform)
↓
Layer 1 (uniform or patterned)
↓
Layer 2 (uniform or patterned)
↓
...
↓
Layer N (uniform or patterned)
↓
Output region (uniform)
```

Each layer is solved independently, then coupled via boundary conditions.

## Truncation and Convergence

### Fourier Truncation

In practice, we truncate the infinite Fourier series:

$$
\mathbf{E}(\mathbf{r}) = \sum_{m=-M}^{M} \sum_{n=-N}^{N} \mathbf{E}_{mn}(z) e^{i\mathbf{k}_{mn} \cdot \mathbf{r}_\parallel}
$$

The truncation order $N_G = (2M+1)(2N+1)$ determines accuracy and computational cost.

### Convergence Behavior

**Smooth features**: Fast convergence, small $N_G$ sufficient

**Sharp features**: Slow convergence, large $N_G$ needed (Gibbs phenomenon)

**Deep subwavelength**: Very large $N_G$ may be required

**Rule of thumb**: Start with $N_G \sim 101$, increase until results converge

## Physical Insights

### Why RCWA Works

1. **Periodicity reduces dimensionality**: 3D problem → unit cell + Bloch theorem
2. **Fourier representation**: Efficient for smooth dielectric profiles
3. **Layer-wise**: Complex 3D structure → stack of 2D problems
4. **Rigorous**: Solves Maxwell's equations exactly (within truncation)

### When RCWA Excels

- **High-Q resonances**: Captures sharp spectral features
- **Subwavelength structures**: Handles near-field correctly
- **Multiple diffraction orders**: Analyzes each order separately
- **Arbitrary patterns**: No restriction on shape (with grid method)

### Limitations

- **Requires periodicity**: Can't simulate isolated objects directly (use super-cell)
- **Convergence for sharp edges**: May need high truncation order
- **Frequency domain**: Time-domain requires Fourier transform

## Summary

RCWA solves Maxwell's equations for periodic structures by:

1. Expanding fields in Floquet-Bloch modes (diffraction orders)
2. Fourier transforming the dielectric profile
3. Solving eigenvalue problems for each layer
4. Matching boundary conditions between layers
5. Computing power flow in each diffraction order

The result is a rigorous, efficient method for simulating periodic photonic structures.

## Next Topics

- **[Mathematical Formulation](mathematics.md)**: Detailed equations and derivations
- **[RCWA Algorithm](algorithm.md)**: Step-by-step computational procedure
- **[Basic Concepts](../guide/concepts.md)**: Practical implementation guide
