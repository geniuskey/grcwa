#!/usr/bin/env python3
"""
Generate diagrams for GRCWA documentation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Set output directory
output_dir = 'docs_mkdocs/assets/images'
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.unicode_minus'] = False

def generate_lattice_structure():
    """Generate real space lattice diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Lattice vectors
    L1 = np.array([1.5, 0.5])
    L2 = np.array([0.3, 1.2])

    # Draw lattice points
    for i in range(-2, 3):
        for j in range(-2, 3):
            point = i * L1 + j * L2
            if np.linalg.norm(point) < 3.5:
                ax.plot(point[0], point[1], 'ko', markersize=8)

    # Origin point (highlighted)
    ax.plot(0, 0, 'ro', markersize=12, label='Origin', zorder=5)

    # Draw lattice vectors from origin
    ax.arrow(0, 0, L1[0], L1[1], head_width=0.15, head_length=0.15,
             fc='blue', ec='blue', linewidth=2.5, label='$\\mathbf{L}_1$', zorder=4)
    ax.arrow(0, 0, L2[0], L2[1], head_width=0.15, head_length=0.15,
             fc='green', ec='green', linewidth=2.5, label='$\\mathbf{L}_2$', zorder=4)

    # Draw unit cell
    vertices = np.array([[0, 0], L1, L1 + L2, L2, [0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], 'r--', linewidth=2, alpha=0.7, label='Unit Cell')
    ax.fill(vertices[:-1, 0], vertices[:-1, 1], alpha=0.1, color='red')

    # Add text annotations
    ax.text(L1[0]/2 - 0.2, L1[1]/2 + 0.3, '$\\mathbf{L}_1$', fontsize=14, color='blue', fontweight='bold')
    ax.text(L2[0]/2 - 0.4, L2[1]/2, '$\\mathbf{L}_2$', fontsize=14, color='green', fontweight='bold')

    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_title('Real Space Lattice', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(-3, 4)
    ax.set_ylim(-2, 3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/lattice_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/lattice_structure.png")


def generate_reciprocal_lattice():
    """Generate reciprocal lattice diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Reciprocal lattice vectors (example)
    G1 = np.array([1.0, -0.3])
    G2 = np.array([-0.2, 1.1])

    # Draw reciprocal lattice points
    for i in range(-3, 4):
        for j in range(-3, 4):
            point = i * G1 + j * G2
            if np.linalg.norm(point) < 4.0:
                ax.plot(point[0], point[1], 'ko', markersize=6)
                # Label diffraction orders near origin
                if abs(i) <= 1 and abs(j) <= 1:
                    ax.text(point[0] + 0.1, point[1] + 0.15, f'({i},{j})',
                           fontsize=9, color='darkblue')

    # Origin (0,0 order - highlighted)
    ax.plot(0, 0, 'ro', markersize=12, label='(0,0) order', zorder=5)

    # Draw reciprocal lattice vectors
    ax.arrow(0, 0, G1[0], G1[1], head_width=0.15, head_length=0.15,
             fc='purple', ec='purple', linewidth=2.5, label='$\\mathbf{G}_1$', zorder=4)
    ax.arrow(0, 0, G2[0], G2[1], head_width=0.15, head_length=0.15,
             fc='orange', ec='orange', linewidth=2.5, label='$\\mathbf{G}_2$', zorder=4)

    # Draw truncation circle (nG example)
    circle = plt.Circle((0, 0), 2.5, color='red', fill=False, linewidth=2,
                        linestyle='--', label='Truncation (nG)', alpha=0.7)
    ax.add_patch(circle)

    ax.text(G1[0]/2 + 0.2, G1[1]/2 - 0.3, '$\\mathbf{G}_1$', fontsize=14,
            color='purple', fontweight='bold')
    ax.text(G2[0]/2 - 0.4, G2[1]/2 + 0.1, '$\\mathbf{G}_2$', fontsize=14,
            color='orange', fontweight='bold')

    ax.set_xlabel('$k_x$', fontsize=14)
    ax.set_ylabel('$k_y$', fontsize=14)
    ax.set_title('Reciprocal Lattice & Diffraction Orders', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/reciprocal_lattice.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/reciprocal_lattice.png")


def generate_layer_stack():
    """Generate layer stack structure diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Layer definitions: (y_start, thickness, color, label, epsilon)
    layers = [
        (5.0, 1.5, '#87CEEB', 'Incident Medium (Air)\n$\\varepsilon = 1.0$', 1.0),
        (3.5, 0.8, '#FFD700', 'Layer 1\n$\\varepsilon(x,y)$', None),
        (2.7, 0.5, '#90EE90', 'Layer 2 (Uniform)\n$\\varepsilon = 2.25$', 2.25),
        (2.2, 1.0, '#FFA07A', 'Layer 3\n$\\varepsilon(x,y)$', None),
        (1.2, 1.2, '#87CEEB', 'Transmission Medium\n$\\varepsilon = 2.1$', 2.1),
    ]

    width = 6
    x_start = 1

    for i, (y_start, thickness, color, label, eps) in enumerate(layers):
        rect = Rectangle((x_start, y_start), width, thickness,
                         facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Add label
        y_center = y_start + thickness / 2
        ax.text(x_start + width/2, y_center, label,
               ha='center', va='center', fontsize=11, fontweight='bold')

        # Add pattern for patterned layers
        if eps is None and i in [1, 3]:
            n_squares = 8
            square_size = width / n_squares / 2
            for j in range(n_squares):
                for k in range(int(thickness / square_size)):
                    if (j + k) % 2 == 0:
                        sq = Rectangle((x_start + j * width/n_squares,
                                      y_start + k * square_size),
                                     width/n_squares, square_size,
                                     facecolor='white', alpha=0.3)
                        ax.add_patch(sq)

    # Draw incident wave
    arrow_x = x_start - 0.5
    arrow_y = 6.8
    ax.annotate('', xy=(arrow_x, 5.2), xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(arrow_x - 0.5, 6.0, 'Incident\n$\\theta, \\phi$', fontsize=11,
            color='red', fontweight='bold')

    # Draw reflected wave
    ax.annotate('', xy=(arrow_x, arrow_y), xytext=(arrow_x, 5.2),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='blue', alpha=0.7))
    ax.text(arrow_x - 0.8, 5.8, 'R', fontsize=12, color='blue', fontweight='bold')

    # Draw transmitted wave
    arrow_x_t = x_start + width + 0.5
    ax.annotate('', xy=(arrow_x_t, 0.0), xytext=(arrow_x_t, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green', alpha=0.7))
    ax.text(arrow_x_t + 0.3, 0.5, 'T', fontsize=12, color='green', fontweight='bold')

    # Add thickness annotations
    for i, (y_start, thickness, _, _, _) in enumerate(layers[1:-1], 1):
        ax.plot([x_start + width + 0.2, x_start + width + 0.2],
               [y_start, y_start + thickness], 'k-', linewidth=1.5)
        ax.text(x_start + width + 0.5, y_start + thickness/2,
               f'd$_{i}$', fontsize=11, va='center')

    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multi-Layer Structure', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_stack.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/layer_stack.png")


def generate_bayer_pattern():
    """Generate Bayer pattern array visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Create 4x4 Bayer pattern
    n = 4
    pattern = [
        ['G', 'R', 'G', 'R'],
        ['B', 'G', 'B', 'G'],
        ['G', 'R', 'G', 'R'],
        ['B', 'G', 'B', 'G']
    ]

    colors = {'R': '#FF6B6B', 'G': '#95E1D3', 'B': '#6B9CFF'}

    for i in range(n):
        for j in range(n):
            color_key = pattern[i][j]
            rect = Rectangle((j, n-1-i), 1, 1,
                           facecolor=colors[color_key],
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(j + 0.5, n-1-i + 0.5, color_key,
                   ha='center', va='center', fontsize=18,
                   fontweight='bold', color='white')

    # Highlight 2x2 unit cell
    rect_unit = Rectangle((0, 2), 2, 2,
                          facecolor='none', edgecolor='yellow',
                          linewidth=4, linestyle='--')
    ax.add_patch(rect_unit)
    ax.text(1, 4.3, '2×2 Unit Cell', ha='center', fontsize=13,
           fontweight='bold', color='yellow',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.2, n + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Bayer Pattern Array (2×2 Unit Cell)', fontsize=16, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Red'),
        patches.Patch(facecolor='#95E1D3', edgecolor='black', label='Green'),
        patches.Patch(facecolor='#6B9CFF', edgecolor='black', label='Blue')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/bayer_pattern.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/bayer_pattern.png")


def generate_sensor_structure():
    """Generate image sensor cross-section structure diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Layer structure (y_bottom, height, color, label)
    x_start = 1
    width = 8

    layers_data = [
        (0, 1.5, '#FF6B6B', 'Si substrate'),
        (1.5, 2.0, '#4A90E2', 'Photodiode\n(charge collection)'),
        (3.5, 0.3, '#999999', 'Metal wiring'),
        (3.8, 0.5, '#95E1D3', 'Color filter'),
        (4.3, 0.8, '#87CEEB', 'Microlens'),
    ]

    for y_bottom, height, color, label in layers_data:
        rect = Rectangle((x_start, y_bottom), width, height,
                        facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)

        # Add text
        y_center = y_bottom + height / 2
        ax.text(x_start + width/2, y_center, label,
               ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw microlens shape on top
    lens_x = np.linspace(x_start, x_start + width, 100)
    lens_radius = width / 2
    lens_curve_height = 0.6
    lens_y = 4.3 + lens_curve_height * np.sqrt(1 - ((lens_x - (x_start + width/2)) / lens_radius)**2)
    ax.fill_between(lens_x, 4.3 + 0.8, lens_y, color='#87CEEB', alpha=0.6, edgecolor='black', linewidth=2)

    # Draw incident light rays
    for x_offset in [2.5, 4.5, 6.5]:
        # Incident ray
        ax.plot([x_start + x_offset, x_start + x_offset], [6.5, 5.5],
               'r-', linewidth=2, alpha=0.7)
        ax.arrow(x_start + x_offset, 5.5, 0, -0.3,
                head_width=0.2, head_length=0.1, fc='red', ec='red', alpha=0.7)

        # Refracted ray through lens (simplified)
        target_x = x_start + width/2
        ax.plot([x_start + x_offset, target_x], [5.1, 2.5],
               'orange', linewidth=2, linestyle='--', alpha=0.7)

    ax.text(x_start + width/2, 7.0, 'Incident Light', ha='center', fontsize=13,
           fontweight='bold', color='red')

    # Add dimension arrows
    ax.annotate('', xy=(x_start + width + 0.5, 0), xytext=(x_start + width + 0.5, 1.5),
               arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(x_start + width + 1.0, 0.75, '~0.5-1μm', fontsize=10, va='center')

    ax.annotate('', xy=(x_start + width + 0.5, 1.5), xytext=(x_start + width + 0.5, 3.5),
               arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(x_start + width + 1.0, 2.5, '~2-3μm', fontsize=10, va='center')

    # Add pixel pitch annotation
    ax.annotate('', xy=(x_start, -0.5), xytext=(x_start + width, -0.5),
               arrowprops=dict(arrowstyle='<->', lw=2))
    ax.text(x_start + width/2, -0.9, 'Pixel pitch ~1-2μm',
           fontsize=11, ha='center', fontweight='bold')

    ax.set_xlim(0, 11)
    ax.set_ylim(-1.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Image Sensor Pixel Structure (Cross-section)', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sensor_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/sensor_structure.png")


def generate_incidence_geometry():
    """Generate incidence angle and polarization geometry diagram"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw coordinate axes
    axis_length = 1.5
    ax.plot([0, axis_length], [0, 0], [0, 0], 'k-', linewidth=2)
    ax.plot([0, 0], [0, axis_length], [0, 0], 'k-', linewidth=2)
    ax.plot([0, 0], [0, 0], [0, axis_length], 'k-', linewidth=2)

    ax.text(axis_length + 0.1, 0, 0, 'x', fontsize=14, fontweight='bold')
    ax.text(0, axis_length + 0.1, 0, 'y', fontsize=14, fontweight='bold')
    ax.text(0, 0, axis_length + 0.1, 'z', fontsize=14, fontweight='bold')

    # Draw surface plane (xy-plane)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightblue')

    # Incidence angles
    theta = np.radians(30)  # polar angle
    phi = np.radians(45)    # azimuthal angle

    # Incident wave vector direction
    k_length = 1.2
    kx = k_length * np.sin(theta) * np.cos(phi)
    ky = k_length * np.sin(theta) * np.sin(phi)
    kz = -k_length * np.cos(theta)  # negative z direction (incident from above)

    # Draw incident wave vector
    ax.quiver(0, 0, 1.5, kx, ky, kz, color='red', arrow_length_ratio=0.15,
             linewidth=3, label='Incident $\\mathbf{k}$')

    # Draw projection on xy-plane
    ax.plot([0, kx], [0, ky], [0, 0], 'r--', linewidth=2, alpha=0.5)
    ax.plot([kx, kx], [ky, ky], [0, 1.5+kz], 'r--', linewidth=1, alpha=0.5)

    # Draw theta angle arc
    theta_arc = np.linspace(np.pi/2, np.pi/2 + theta, 20)
    arc_r = 0.4
    arc_x = arc_r * np.sin(theta_arc) * np.cos(phi)
    arc_y = arc_r * np.sin(theta_arc) * np.sin(phi)
    arc_z = 1.5 + arc_r * np.cos(theta_arc)
    ax.plot(arc_x, arc_y, arc_z, 'b-', linewidth=2)
    ax.text(arc_x[10], arc_y[10], arc_z[10] - 0.2, '$\\theta$', fontsize=14,
           color='blue', fontweight='bold')

    # Draw phi angle arc (in xy-plane)
    phi_arc = np.linspace(0, phi, 20)
    arc_r2 = 0.5
    arc_x2 = arc_r2 * np.cos(phi_arc)
    arc_y2 = arc_r2 * np.sin(phi_arc)
    arc_z2 = np.zeros_like(phi_arc)
    ax.plot(arc_x2, arc_y2, arc_z2, 'g-', linewidth=2)
    ax.text(arc_x2[10] + 0.1, arc_y2[10], 0.1, '$\\phi$', fontsize=14,
           color='green', fontweight='bold')

    # Draw polarization vectors
    # s-polarization (perpendicular to plane of incidence)
    s_pol = np.array([-np.sin(phi), np.cos(phi), 0])
    s_pol = s_pol / np.linalg.norm(s_pol) * 0.6
    origin_pol = np.array([kx/2, ky/2, 1.5 + kz/2])
    ax.quiver(origin_pol[0], origin_pol[1], origin_pol[2],
             s_pol[0], s_pol[1], s_pol[2],
             color='purple', arrow_length_ratio=0.2, linewidth=2.5,
             label='s-pol (TE)')

    # p-polarization (in plane of incidence)
    p_pol = np.cross([kx, ky, kz], s_pol)
    p_pol = p_pol / np.linalg.norm(p_pol) * 0.6
    ax.quiver(origin_pol[0], origin_pol[1], origin_pol[2],
             p_pol[0], p_pol[1], p_pol[2],
             color='orange', arrow_length_ratio=0.2, linewidth=2.5,
             label='p-pol (TM)')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Incidence Geometry & Polarization', fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim([-1, 1.5])
    ax.set_ylim([-1, 1.5])
    ax.set_zlim([0, 2])

    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/incidence_geometry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/incidence_geometry.png")


def generate_diffraction_orders():
    """Generate diffraction orders visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Draw grating surface
    x_grating = np.linspace(-5, 5, 200)
    y_grating = np.zeros_like(x_grating)
    ax.plot(x_grating, y_grating, 'k-', linewidth=3)

    # Draw grating structure (periodic)
    period = 1.0
    n_periods = 10
    for i in range(-n_periods//2, n_periods//2 + 1):
        x_start = i * period - period/4
        rect = Rectangle((x_start, -0.3), period/2, 0.3,
                        facecolor='gray', edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)

    # Incident beam
    incident_angle = np.radians(20)
    beam_x = 0
    beam_length = 2
    ax.arrow(beam_x - beam_length * np.sin(incident_angle),
            beam_length * np.cos(incident_angle),
            beam_length * np.sin(incident_angle),
            -beam_length * np.cos(incident_angle),
            head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2.5)
    ax.text(beam_x - 1.2, 1.5, 'Incident\n$\\theta_i$', fontsize=12, color='red', fontweight='bold')

    # Diffraction orders
    orders = [-2, -1, 0, 1, 2]
    colors = ['blue', 'cyan', 'green', 'orange', 'purple']

    lambda_wave = 0.6  # wavelength in units of period
    for m, color in zip(orders, colors):
        # Grating equation: sin(theta_m) = sin(theta_i) + m * lambda / period
        sin_theta_m = np.sin(incident_angle) + m * lambda_wave / period

        if abs(sin_theta_m) <= 1:  # Propagating order
            theta_m = np.arcsin(sin_theta_m)

            # Draw diffracted beam
            beam_x_start = 0
            beam_y_start = 0
            beam_length_diff = 2.5

            ax.arrow(beam_x_start, beam_y_start,
                    beam_length_diff * np.sin(theta_m),
                    -beam_length_diff * np.cos(theta_m) if m < 0 else beam_length_diff * np.cos(theta_m),
                    head_width=0.1, head_length=0.08, fc=color, ec=color,
                    linewidth=2, alpha=0.7, linestyle='-' if m == 0 else '--')

            # Label
            label_x = beam_x_start + 1.8 * np.sin(theta_m)
            label_y = -1.8 * np.cos(theta_m) if m < 0 else 1.8 * np.cos(theta_m)
            ax.text(label_x, label_y, f'm={m}', fontsize=11, color=color, fontweight='bold')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Diffraction Orders', fontsize=16, fontweight='bold', pad=20)

    # Add legend
    ax.text(2.5, 2.5, 'Propagating\nOrders', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/diffraction_orders.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/diffraction_orders.png")


def generate_fourier_expansion():
    """Generate Fourier series expansion visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Original square wave function
    x = np.linspace(0, 2*np.pi, 1000)
    def square_wave(x):
        return np.where(np.mod(x, 2*np.pi) < np.pi, 1, -1)

    original = square_wave(x)

    # Different numbers of Fourier terms
    n_terms_list = [1, 3, 7, 15]

    for idx, n_terms in enumerate(n_terms_list):
        ax = axes[idx]

        # Compute Fourier series approximation
        fourier_approx = np.zeros_like(x)
        for n in range(1, n_terms + 1, 2):  # Only odd harmonics for square wave
            fourier_approx += (4 / (n * np.pi)) * np.sin(n * x)

        # Plot
        ax.plot(x, original, 'k-', linewidth=2, label='Original', alpha=0.3)
        ax.plot(x, fourier_approx, 'b-', linewidth=2.5, label=f'Fourier (n={n_terms})')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('f(x)', fontsize=12)
        ax.set_title(f'N = {n_terms} terms', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(-1.5, 1.5)

    fig.suptitle('Fourier Series Convergence', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fourier_expansion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/fourier_expansion.png")


def generate_propagating_evanescent():
    """Generate propagating vs evanescent modes visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Propagating mode (real kz)
    z = np.linspace(0, 5, 200)
    kz_real = 2.0
    wave_prop = np.exp(1j * kz_real * z)

    ax1.plot(z, np.real(wave_prop), 'b-', linewidth=2.5, label='Re[E]')
    ax1.plot(z, np.imag(wave_prop), 'r--', linewidth=2.5, label='Im[E]')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('z (depth)', fontsize=13)
    ax1.set_ylabel('Electric Field', fontsize=13)
    ax1.set_title('Propagating Mode (Real $k_z$)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.text(2.5, 0.7, '$k_z = $ real\nCarries power', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Evanescent mode (imaginary kz)
    kz_imag = 1.5
    wave_evan = np.exp(-kz_imag * z)

    ax2.plot(z, wave_evan, 'b-', linewidth=2.5, label='|E|')
    ax2.fill_between(z, 0, wave_evan, alpha=0.3, color='blue')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('z (depth)', fontsize=13)
    ax2.set_ylabel('Electric Field Magnitude', fontsize=13)
    ax2.set_title('Evanescent Mode (Imaginary $k_z$)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.text(2.5, 0.5, '$k_z = $ imaginary\nExponential decay\nNo power flow', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/propagating_evanescent.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/propagating_evanescent.png")


def generate_boundary_conditions():
    """Generate boundary conditions illustration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Tangential components (continuous)
    y = np.linspace(-2, 2, 100)

    # Interface at x=0
    E_tang_left = np.ones_like(y[y < 0]) * 1.0
    E_tang_right = np.ones_like(y[y >= 0]) * 1.0

    ax1.axvline(x=0, color='k', linewidth=3, label='Interface')
    ax1.fill_betweenx(y, -2, 0, alpha=0.2, color='blue', label='Medium 1')
    ax1.fill_betweenx(y, 0, 2, alpha=0.2, color='red', label='Medium 2')

    # Draw tangential field (continuous)
    ax1.plot([-1.5, -0.05], [0, 0], 'b-', linewidth=3)
    ax1.plot([0.05, 1.5], [0, 0], 'r-', linewidth=3)
    ax1.plot([-0.05, 0.05], [0, 0], 'g-', linewidth=4, label='$E_{\\parallel}$ continuous')

    # Arrows showing field direction
    ax1.arrow(-1, 0, 0.3, 0, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
    ax1.arrow(1, 0, 0.3, 0, head_width=0.2, head_length=0.1, fc='red', ec='red')

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('y', fontsize=13)
    ax1.set_title('Tangential Component:\n$\\mathbf{n} \\times (\\mathbf{E}_1 - \\mathbf{E}_2) = 0$',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_aspect('equal')

    # Normal components (discontinuous for E, continuous for D)
    ax2.axvline(x=0, color='k', linewidth=3, label='Interface')
    ax2.fill_betweenx(y, -2, 0, alpha=0.2, color='blue', label='$\\varepsilon_1 = 1$')
    ax2.fill_betweenx(y, 0, 2, alpha=0.2, color='red', label='$\\varepsilon_2 = 4$')

    # Draw normal field (discontinuous in E)
    E1_normal = 1.0
    E2_normal = 0.25  # D continuous, so E2 = D/eps2 = E1*eps1/eps2

    ax2.arrow(-1, -1.5, 0, E1_normal*0.8, head_width=0.15, head_length=0.1,
             fc='blue', ec='blue', linewidth=2.5, label=f'$E_1^\\perp$ = {E1_normal}')
    ax2.arrow(1, -1.5, 0, E2_normal*0.8, head_width=0.15, head_length=0.1,
             fc='red', ec='red', linewidth=2.5, label=f'$E_2^\\perp$ = {E2_normal}')

    ax2.text(-1, 1, '$D_1^\\perp = \\varepsilon_1 E_1^\\perp = 1.0$', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.text(0.3, 1, '$D_2^\\perp = \\varepsilon_2 E_2^\\perp = 1.0$', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('z', fontsize=13)
    ax2.set_title('Normal Component:\n$\\mathbf{n} \\cdot (\\varepsilon_1\\mathbf{E}_1 - \\varepsilon_2\\mathbf{E}_2) = 0$',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/boundary_conditions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/boundary_conditions.png")


def generate_eigenmode_concept():
    """Generate eigenmode concept diagram"""
    fig = plt.figure(figsize=(12, 8))

    # Create 3D plot
    ax = fig.add_subplot(111, projection='3d')

    # Create layer
    z_layer = np.array([0, 0, 1, 1])
    x_layer = np.array([0, 2, 2, 0])
    y_layer = np.array([0, 0, 0, 0])

    # Draw multiple layers to show the structure
    for i in range(3):
        z_offset = i * 1.5
        verts = [list(zip([0, 2, 2, 0], [0, 0, 1, 1], [z_offset]*4)),
                 list(zip([0, 2, 2, 0], [0, 0, 1, 1], [z_offset+1]*4))]

        # Draw top and bottom faces
        for vert in verts:
            poly = Poly3DCollection([vert], alpha=0.3, facecolor='lightblue',
                                   edgecolor='black', linewidth=2)
            ax.add_collection3d(poly)

    # Draw eigenmodes (sinusoidal patterns)
    x = np.linspace(0, 2, 50)
    z = np.linspace(0, 4, 100)

    # Mode 1: forward propagating
    for i, z_val in enumerate(np.linspace(0.5, 3.5, 6)):
        y_wave = 0.3 * np.sin(2 * np.pi * x) + 0.5
        z_wave = np.ones_like(x) * z_val
        ax.plot(x, y_wave, z_wave, 'r-', linewidth=2, alpha=0.7)

    ax.text(1, 0.8, 2, 'Eigenmodes\n$\\psi_n(x,y) e^{ik_{z,n}z}$', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Add arrows showing propagation
    ax.quiver(0.5, 0.5, 1, 0, 0, 1, color='red', arrow_length_ratio=0.3,
             linewidth=2.5, label='Forward mode')
    ax.quiver(1.5, 0.5, 3, 0, 0, -1, color='blue', arrow_length_ratio=0.3,
             linewidth=2.5, label='Backward mode')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z (propagation)', fontsize=12)
    ax.set_title('Eigenmode Propagation in Patterned Layer', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/eigenmode_concept.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/eigenmode_concept.png")


def generate_convergence_test():
    """Generate convergence test visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Simulate convergence data
    nG_values = np.array([11, 21, 51, 101, 151, 201, 301, 401, 501])

    # Reflection converges faster (smoother structure)
    R_values = 0.35 + 0.15 * np.exp(-nG_values / 80) + 0.02 * np.random.randn(len(nG_values))
    R_values = np.clip(R_values, 0, 1)

    # Transmission
    T_values = 1 - R_values - 0.05 * np.exp(-nG_values / 100)
    T_values = np.clip(T_values, 0, 1)

    ax1.plot(nG_values, R_values, 'bo-', linewidth=2.5, markersize=8, label='Reflectance (R)')
    ax1.plot(nG_values, T_values, 'rs-', linewidth=2.5, markersize=8, label='Transmittance (T)')
    ax1.plot(nG_values, R_values + T_values, 'g^--', linewidth=2, markersize=7,
            label='R + T (Energy conservation)')
    ax1.axhline(y=1, color='k', linestyle=':', alpha=0.5, label='Perfect conservation')

    ax1.set_xlabel('Number of Fourier orders (nG)', fontsize=13)
    ax1.set_ylabel('Value', fontsize=13)
    ax1.set_title('Convergence with nG', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Computation time vs nG
    time_values = (nG_values / 11) ** 2.5 * 0.1  # Roughly O(nG^2.5)

    ax2.loglog(nG_values, time_values, 'mo-', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Number of Fourier orders (nG)', fontsize=13)
    ax2.set_ylabel('Computation Time (s)', fontsize=13)
    ax2.set_title('Computational Cost', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.text(150, 50, 'Scaling: $\\sim O(nG^{2.5})$', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/convergence_test.png")


def generate_smatrix_concept():
    """Generate S-matrix concept diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Draw the structure (box representing the system)
    structure = Rectangle((3, 3), 4, 3, facecolor='lightblue',
                         edgecolor='black', linewidth=3, alpha=0.5)
    ax.add_patch(structure)
    ax.text(5, 4.5, 'RCWA\nStructure', ha='center', fontsize=14, fontweight='bold')

    # Input fields (left side)
    # Forward propagating (incident)
    ax.arrow(0.5, 5, 2, 0, head_width=0.3, head_length=0.3,
            fc='red', ec='red', linewidth=3)
    ax.text(1.5, 5.7, '$a^+$ (Incident)', fontsize=12, color='red', fontweight='bold')

    # Backward propagating (from right, entering left)
    ax.arrow(2.5, 3.5, -2, 0, head_width=0.3, head_length=0.3,
            fc='blue', ec='blue', linewidth=2, alpha=0.7)
    ax.text(1.5, 2.8, '$a^-$', fontsize=12, color='blue', fontweight='bold')

    # Output fields (left side)
    # Reflected (backward propagating)
    ax.arrow(2.5, 5.5, -2, 0, head_width=0.3, head_length=0.3,
            fc='green', ec='green', linewidth=2.5)
    ax.text(0.8, 6.2, '$b^-$ (Reflected)', fontsize=12, color='green', fontweight='bold')

    # Output fields (right side)
    # Transmitted (forward propagating)
    ax.arrow(7.5, 4.5, 2, 0, head_width=0.3, head_length=0.3,
            fc='orange', ec='orange', linewidth=2.5)
    ax.text(8, 5.2, '$b^+$ (Transmitted)', fontsize=12, color='orange', fontweight='bold')

    # Backward propagating (from right)
    ax.arrow(9.5, 4, -2, 0, head_width=0.3, head_length=0.3,
            fc='purple', ec='purple', linewidth=2, alpha=0.7)
    ax.text(8.5, 3.3, '$a^-$', fontsize=12, color='purple', fontweight='bold')

    # S-matrix equations (simplified without pmatrix)
    eq_text = (
        'S-Matrix Relation:\n\n'
        '[b⁻]   [S₁₁  S₁₂]  [a⁺]\n'
        '[b⁺] = [S₂₁  S₂₂]  [a⁻]\n\n'
        'S₁₁: Reflection from left\n'
        'S₂₁: Transmission left→right\n'
        'S₁₂: Transmission right→left\n'
        'S₂₂: Reflection from right'
    )

    ax.text(5, 1, eq_text, ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('S-Matrix Formulation', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/smatrix_concept.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/smatrix_concept.png")


# Generate all figures
if __name__ == '__main__':
    print("Generating documentation figures...")
    print()

    # Basic structural diagrams
    generate_lattice_structure()
    generate_reciprocal_lattice()
    generate_layer_stack()
    generate_bayer_pattern()
    generate_sensor_structure()
    generate_incidence_geometry()
    generate_diffraction_orders()

    # Theory concept diagrams
    generate_fourier_expansion()
    generate_propagating_evanescent()
    generate_boundary_conditions()
    generate_eigenmode_concept()
    generate_convergence_test()
    generate_smatrix_concept()

    print()
    print(f"All figures generated successfully in {output_dir}/")
