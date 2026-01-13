# Mathematical Formulation of RCWA

This page derives the mathematical framework of RCWA in detail.

## Field Expansion

### Fourier Series Expansion

For a periodic structure with lattice vectors $\mathbf{L}_1, \mathbf{L}_2$, any periodic function can be expanded:

$$
f(\mathbf{r}_\parallel) = \sum_{m,n} f_{mn} \exp(i\mathbf{G}_{mn} \cdot \mathbf{r}_\parallel)
$$

where $\mathbf{G}_{mn} = m\mathbf{K}_1 + n\mathbf{K}_2$ are reciprocal lattice vectors.

### Field Components

The electric and magnetic fields in each layer are:

$$
\mathbf{E}(\mathbf{r}) = \sum_{mn} \mathbf{E}_{mn}(z) \exp(i\mathbf{k}_{mn,\parallel} \cdot \mathbf{r}_\parallel)
$$

$$
\mathbf{H}(\mathbf{r}) = \sum_{mn} \mathbf{H}_{mn}(z) \exp(i\mathbf{k}_{mn,\parallel} \cdot \mathbf{r}_\parallel)
$$

where $\mathbf{k}_{mn,\parallel} = \mathbf{k}_{\parallel,0} + \mathbf{G}_{mn}$.

## Eigenvalue Problem in Patterned Layers

### Maxwell's Equations in Fourier Space

Applying Fourier transform to curl equations:

$$
\nabla \times \mathbf{E} = i\omega\mu\mathbf{H}
$$

$$
\nabla \times \mathbf{H} = -i\omega\varepsilon\mathbf{E}
$$

For TM modes (p-polarization), we eliminate $\mathbf{H}$ to get:

$$
\nabla \times (\varepsilon^{-1} \nabla \times \mathbf{H}) = \omega^2 \mu \mathbf{H}
$$

For TE modes (s-polarization):

$$
\nabla \times (\mu^{-1} \nabla \times \mathbf{E}) = \omega^2 \varepsilon \mathbf{E}
$$

### Tangential Field Formulation

We work with tangential field components $(E_x, E_y, H_x, H_y)$ since they are continuous across interfaces.

Define vectors:

$$
\mathbf{E}_\parallel = \begin{pmatrix} E_x \\ E_y \end{pmatrix}, \quad
\mathbf{H}_\parallel = \begin{pmatrix} H_x \\ H_y \end{pmatrix}
$$

From Maxwell's equations:

$$
\frac{\partial}{\partial z} \mathbf{E}_\parallel = i\omega\mu \hat{z} \times \mathbf{H}_\parallel - i\mathbf{k}_\parallel E_z
$$

$$
\frac{\partial}{\partial z} \mathbf{H}_\parallel = -i\omega\varepsilon \hat{z} \times \mathbf{E}_\parallel - i\mathbf{k}_\parallel H_z
$$

### Eliminating Normal Components

From $\nabla \cdot \mathbf{D} = 0$:

$$
i k_x E_x + i k_y E_y + \frac{\partial (\varepsilon E_z)}{\partial z} = 0
$$

For z-invariant $\varepsilon$ in each layer:

$$
E_z = -\frac{1}{\varepsilon}(k_x E_x + k_y E_y)
$$

Similarly from $\nabla \cdot \mathbf{B} = 0$:

$$
H_z = -\frac{1}{\mu}(k_x H_x + k_y H_y)
$$

### Matrix Formulation

Define the $K_\perp$ operator:

$$
K_\perp = \begin{pmatrix} k_y^2 & -k_x k_y \\ -k_x k_y & k_x^2 \end{pmatrix}
$$

This acts on the Fourier space with convolution matrices.

The coupled wave equations become:

$$
\frac{\partial}{\partial z} \begin{pmatrix} \mathbf{E}_\parallel \\ \mathbf{H}_\parallel \end{pmatrix}
= i \begin{pmatrix} 0 & A \\ B & 0 \end{pmatrix}
\begin{pmatrix} \mathbf{E}_\parallel \\ \mathbf{H}_\parallel \end{pmatrix}
$$

where $A$ and $B$ are matrices involving $\varepsilon$, $\mu$, and $K_\perp$.

### Eigenvalue Problem

Substituting $\exp(iq z)$ dependence:

$$
\begin{pmatrix} \mathbf{E}_\parallel \\ \mathbf{H}_\parallel \end{pmatrix}
= \begin{pmatrix} \boldsymbol{\phi}_E \\ \boldsymbol{\phi}_H \end{pmatrix} e^{iq z}
$$

This gives the eigenvalue equation:

$$
\begin{pmatrix} 0 & A \\ B & 0 \end{pmatrix}
\begin{pmatrix} \boldsymbol{\phi}_E \\ \boldsymbol{\phi}_H \end{pmatrix}
= q \begin{pmatrix} \boldsymbol{\phi}_E \\ \boldsymbol{\phi}_H \end{pmatrix}
$$

Or equivalently:

$$
AB \boldsymbol{\phi}_H = q^2 \boldsymbol{\phi}_H
$$

$$
BA \boldsymbol{\phi}_E = q^2 \boldsymbol{\phi}_E
$$

The eigenvalues $q$ are the $z$-components of wave vectors for eigenmodes.

## Convolution Matrices

### Dielectric Fourier Transform

The dielectric function is expanded:

$$
\varepsilon(x,y) = \sum_{mn} \varepsilon_{mn} \exp(i\mathbf{G}_{mn} \cdot \mathbf{r}_\parallel)
$$

Fourier coefficients:

$$
\varepsilon_{mn} = \frac{1}{A_{\text{cell}}} \int_{\text{cell}} \varepsilon(x,y) \exp(-i\mathbf{G}_{mn} \cdot \mathbf{r}_\parallel) dx dy
$$

In practice, computed via 2D FFT:

$$
\varepsilon_{mn} = \text{FFT2D}[\varepsilon(x,y)]
$$

### Convolution in Fourier Space

Multiplication in real space → convolution in Fourier space:

$$
[\varepsilon(x,y) E_x(x,y,z)]_{mn} = \sum_{m'n'} \varepsilon_{m-m',n-n'} E_{x,m'n'}(z)
$$

This is represented as a matrix multiplication:

$$
(\varepsilon \mathbf{E})_{mn} = \sum_{m'n'} [\mathcal{E}]_{mn,m'n'} E_{m'n'}
$$

where $[\mathcal{E}]_{mn,m'n'} = \varepsilon_{m-m',n-n'}$ is the **convolution matrix**.

### Laurent's Rule for $\varepsilon^{-1}$

For products involving $\varepsilon^{-1}$, direct Fourier transform of $1/\varepsilon(x,y)$ gives:

$$
[\varepsilon^{-1}]_{mn} = \text{FFT2D}[1/\varepsilon(x,y)]
$$

This is **Laurent's rule** and provides better convergence for $\varepsilon^{-1}$ than inverting $\mathcal{E}$.

## S-Matrix Algorithm

### Layer Transfer Matrix

Within a uniform or patterned layer, the solution is:

$$
\begin{pmatrix} \mathbf{E}_\parallel(z) \\ \mathbf{H}_\parallel(z) \end{pmatrix}
= \sum_j c_j^+ \begin{pmatrix} \boldsymbol{\phi}_{E,j} \\ \boldsymbol{\phi}_{H,j} \end{pmatrix} e^{iq_j z}
+ \sum_j c_j^- \begin{pmatrix} \boldsymbol{\phi}_{E,j} \\ -\boldsymbol{\phi}_{H,j} \end{pmatrix} e^{-iq_j z}
$$

where $c_j^+$ and $c_j^-$ are forward and backward mode amplitudes.

At two positions $z_1$ and $z_2 = z_1 + d$:

$$
\begin{pmatrix} \mathbf{E}_\parallel(z_2) \\ \mathbf{H}_\parallel(z_2) \end{pmatrix}
= T \begin{pmatrix} \mathbf{E}_\parallel(z_1) \\ \mathbf{H}_\parallel(z_1) \end{pmatrix}
$$

where $T$ is the transfer matrix.

### S-Matrix Definition

The **scattering matrix** (S-matrix) relates incoming and outgoing waves:

$$
\begin{pmatrix} \mathbf{c}^+_{\text{out}} \\ \mathbf{c}^-_{\text{in}} \end{pmatrix}
= \begin{pmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{pmatrix}
\begin{pmatrix} \mathbf{c}^+_{\text{in}} \\ \mathbf{c}^-_{\text{out}} \end{pmatrix}
$$

where:

- $S_{11}$: Reflection from above
- $S_{22}$: Reflection from below
- $S_{12}$, $S_{21}$: Transmission

### S-Matrix Composition

For two layers with S-matrices $S^{(A)}$ and $S^{(B)}$ stacked in sequence:

$$
S^{(AB)}_{11} = S^{(A)}_{11} + S^{(A)}_{12} (I - S^{(B)}_{11} S^{(A)}_{22})^{-1} S^{(B)}_{11} S^{(A)}_{21}
$$

$$
S^{(AB)}_{12} = S^{(A)}_{12} (I - S^{(B)}_{11} S^{(A)}_{22})^{-1} S^{(B)}_{12}
$$

$$
S^{(AB)}_{21} = S^{(B)}_{21} (I - S^{(A)}_{22} S^{(B)}_{11})^{-1} S^{(A)}_{21}
$$

$$
S^{(AB)}_{22} = S^{(B)}_{22} + S^{(B)}_{21} (I - S^{(A)}_{22} S^{(B)}_{11})^{-1} S^{(A)}_{22} S^{(B)}_{12}
$$

This allows building the total S-matrix layer by layer.

## Boundary Conditions at Input/Output

### Input Region

Semi-infinite uniform region with $\varepsilon_{\text{in}}$:

$$
\mathbf{E}^{\text{in}} = \mathbf{E}_{\text{inc}} e^{i\mathbf{k}_{\text{inc}} \cdot \mathbf{r}}
+ \sum_{mn} r_{mn} \mathbf{E}_{mn}^{\text{refl}} e^{i\mathbf{k}_{mn}^{\text{refl}} \cdot \mathbf{r}}
$$

where:

- $\mathbf{E}_{\text{inc}}$: Incident field
- $r_{mn}$: Reflection coefficients for each order

### Output Region

Semi-infinite uniform region with $\varepsilon_{\text{out}}$:

$$
\mathbf{E}^{\text{out}} = \sum_{mn} t_{mn} \mathbf{E}_{mn}^{\text{trans}} e^{i\mathbf{k}_{mn}^{\text{trans}} \cdot \mathbf{r}}
$$

where $t_{mn}$ are transmission coefficients.

### Solving for Coefficients

Applying boundary conditions at top and bottom interfaces:

$$
\begin{pmatrix} \mathbf{r} \\ \mathbf{t} \end{pmatrix}
= S_{\text{total}} \begin{pmatrix} \mathbf{inc} \\ 0 \end{pmatrix}
$$

where $S_{\text{total}}$ is the total S-matrix of the entire structure.

## Power Computation

### Normalized Poynting Flux

For each diffraction order $(m,n)$, the normalized power is:

$$
P_{mn} = \frac{1}{2} \text{Re}\left( \frac{k_{z,mn}}{\omega\mu} \right) |A_{mn}|^2
$$

where $A_{mn}$ is the field amplitude.

### Reflection and Transmission

Total reflection:

$$
R = \sum_{mn} \frac{k_{z,mn}^{\text{refl}} / \omega\mu_{\text{in}}}{k_{z,\text{inc}} / \omega\mu_{\text{in}}} |r_{mn}|^2
= \sum_{mn} \frac{\text{Re}(k_{z,mn}^{\text{refl}})}{\text{Re}(k_{z,\text{inc}})} |r_{mn}|^2
$$

Total transmission:

$$
T = \sum_{mn} \frac{k_{z,mn}^{\text{trans}} / \omega\mu_{\text{out}}}{k_{z,\text{inc}} / \omega\mu_{\text{in}}} |t_{mn}|^2
= \sum_{mn} \frac{\text{Re}(k_{z,mn}^{\text{trans}}) \mu_{\text{in}}}{\text{Re}(k_{z,\text{inc}}) \mu_{\text{out}}} |t_{mn}|^2
$$

The normalization ensures $R + T = 1$ for lossless structures.

## Field Reconstruction

### Fourier Space Fields

At any position $(x, y, z)$ within a layer, fields are:

$$
\mathbf{E}(x,y,z) = \sum_{mn} \mathbf{E}_{mn}(z) \exp(i k_{x,mn} x + i k_{y,mn} y)
$$

where $\mathbf{E}_{mn}(z)$ are computed from eigenmode expansion.

### Real Space Fields

To get fields on a real-space grid, use inverse FFT:

$$
\mathbf{E}(x_i, y_j, z) = \text{IFFT2D}[\mathbf{E}_{mn}(z)]
$$

### Volume Integrals

For computing volume integrals (e.g., energy density):

$$
\int_V |\mathbf{E}|^2 dV = \sum_{mn} |\mathbf{E}_{mn}|^2 V_{\text{unit}}
$$

where $V_{\text{unit}} = A_{\text{cell}} \cdot d$ is the unit cell volume.

## Numerical Stability

### Enhanced Transmittance Matrix

For thick layers, direct exponentials $e^{iq_j d}$ can overflow for large $|q_j|$.

Use the **Enhanced Transmittance Matrix** method:

- Scale exponentials by largest $q_j$
- Compute ratios rather than absolute values
- Prevents overflow/underflow

### Eigenvalue Ordering

Eigenvalues $q_j$ should be ordered:

- Forward propagating: $\text{Re}(q) > 0$ or $\text{Im}(q) < 0$
- Backward propagating: $\text{Re}(q) < 0$ or $\text{Im}(q) > 0$

This ensures numerical stability in the S-matrix algorithm.

### Matrix Inversion

Matrices like $\mathcal{E}$ can be ill-conditioned for:

- High contrast structures
- Large truncation order
- Thin features

Use regularization or iterative solvers if direct inversion fails.

## Convergence Acceleration

### Adaptive Coordinate Transformation

For gratings with vertical sidewalls, map to $u$-space:

$$
u(x) = x + \sum_n a_n \sin(2\pi n x / \Lambda)
$$

This smooths out discontinuities and improves convergence.

### Perfectly Matched Layer (PML)

For absorbing boundaries (not typically used in periodic RCWA):

$$
\varepsilon \to \varepsilon s, \quad \mu \to \mu s
$$

where $s = 1 + i\sigma/\omega$ is the PML stretching parameter.

### Subpixel Smoothing

For sharp features, average $\varepsilon$ over grid cells rather than point sampling:

$$
\varepsilon_{\text{cell}} = \frac{1}{A_{\text{cell}}} \int_{\text{cell}} \varepsilon(x,y) dx dy
$$

Improves convergence for staircasing approximation.

## Summary of Key Equations

| Quantity | Equation |
|----------|----------|
| **Field expansion** | $\mathbf{E} = \sum_{mn} \mathbf{E}_{mn}(z) e^{i\mathbf{k}_{mn} \cdot \mathbf{r}_\parallel}$ |
| **Wave vector** | $\mathbf{k}_{mn} = \mathbf{k}_{\parallel,0} + \mathbf{G}_{mn}$ |
| **Eigenvalue problem** | $AB\boldsymbol{\phi} = q^2 \boldsymbol{\phi}$ |
| **S-matrix** | $\begin{pmatrix} \mathbf{out} \end{pmatrix} = S \begin{pmatrix} \mathbf{in} \end{pmatrix}$ |
| **Reflection** | $R = \sum_{mn} \frac{\text{Re}(k_{z,mn}^r)}{\text{Re}(k_{z,inc})} |r_{mn}|^2$ |
| **Transmission** | $T = \sum_{mn} \frac{\text{Re}(k_{z,mn}^t)}{\text{Re}(k_{z,inc})} |t_{mn}|^2$ |
| **Energy conservation** | $R + T = 1$ (lossless) |

## Implementation in GRCWA

GRCWA implements these equations in:

- **`rcwa.py`**: Main solver, S-matrix algorithm
- **`fft_funs.py`**: Convolution matrices via FFT
- **`kbloch.py`**: Reciprocal lattice and wave vectors
- **`primitives.py`**: Autograd-compatible eigenvalue solver

Key optimizations:

- Fast FFT for convolution matrices
- Enhanced transmittance matrix for stability
- Autograd primitives for gradient computation

## Further Reading

For derivations and proofs, see:

- Moharam et al., "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings," JOSA A (1995)
- Liu and Fan, "Three-dimensional photonic crystals by total-internal reflection," Optics Letters (2005)
- Whittaker and Culshaw, "Scattering-matrix treatment of patterned multilayer photonic structures," Phys. Rev. B (1999)

## Next Steps

- **[RCWA Algorithm](algorithm.md)**: Step-by-step computational procedure
- **[API Reference](../api/core.md)**: How equations map to code
- **[Tutorials](../tutorials/tutorial1.md)**: Apply the theory in practice
