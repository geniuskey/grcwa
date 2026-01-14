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
    ax.plot(0, 0, 'ro', markersize=12, label='원점', zorder=5)

    # Draw lattice vectors from origin
    ax.arrow(0, 0, L1[0], L1[1], head_width=0.15, head_length=0.15,
             fc='blue', ec='blue', linewidth=2.5, label='$\\mathbf{L}_1$', zorder=4)
    ax.arrow(0, 0, L2[0], L2[1], head_width=0.15, head_length=0.15,
             fc='green', ec='green', linewidth=2.5, label='$\\mathbf{L}_2$', zorder=4)

    # Draw unit cell
    vertices = np.array([[0, 0], L1, L1 + L2, L2, [0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], 'r--', linewidth=2, alpha=0.7, label='단위 셀')
    ax.fill(vertices[:-1, 0], vertices[:-1, 1], alpha=0.1, color='red')

    # Add text annotations
    ax.text(L1[0]/2 - 0.2, L1[1]/2 + 0.3, '$\\mathbf{L}_1$', fontsize=14, color='blue', fontweight='bold')
    ax.text(L2[0]/2 - 0.4, L2[1]/2, '$\\mathbf{L}_2$', fontsize=14, color='green', fontweight='bold')

    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_title('실공간 격자 구조 (Real Space Lattice)', fontsize=15, fontweight='bold')
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
    ax.plot(0, 0, 'ro', markersize=12, label='(0,0) 차수', zorder=5)

    # Draw reciprocal lattice vectors
    ax.arrow(0, 0, G1[0], G1[1], head_width=0.15, head_length=0.15,
             fc='purple', ec='purple', linewidth=2.5, label='$\\mathbf{G}_1$', zorder=4)
    ax.arrow(0, 0, G2[0], G2[1], head_width=0.15, head_length=0.15,
             fc='orange', ec='orange', linewidth=2.5, label='$\\mathbf{G}_2$', zorder=4)

    # Draw truncation circle (nG example)
    circle = plt.Circle((0, 0), 2.5, color='red', fill=False, linewidth=2,
                        linestyle='--', label='절단 영역 (nG)', alpha=0.7)
    ax.add_patch(circle)

    ax.text(G1[0]/2 + 0.2, G1[1]/2 - 0.3, '$\\mathbf{G}_1$', fontsize=14,
            color='purple', fontweight='bold')
    ax.text(G2[0]/2 - 0.4, G2[1]/2 + 0.1, '$\\mathbf{G}_2$', fontsize=14,
            color='orange', fontweight='bold')

    ax.set_xlabel('$k_x$', fontsize=14)
    ax.set_ylabel('$k_y$', fontsize=14)
    ax.set_title('역격자 공간 및 회절 차수 (Reciprocal Lattice)', fontsize=15, fontweight='bold')
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
        (5.0, 1.5, '#87CEEB', '입사 매질 (Air)\n$\\varepsilon = 1.0$', 1.0),
        (3.5, 0.8, '#FFD700', 'Layer 1\n$\\varepsilon(x,y)$', None),
        (2.7, 0.5, '#90EE90', 'Layer 2 (Uniform)\n$\\varepsilon = 2.25$', 2.25),
        (2.2, 1.0, '#FFA07A', 'Layer 3\n$\\varepsilon(x,y)$', None),
        (1.2, 1.2, '#87CEEB', '투과 매질 (Substrate)\n$\\varepsilon = 2.1$', 2.1),
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
    ax.text(arrow_x - 0.5, 6.0, '입사파\n$\\theta, \\phi$', fontsize=11,
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
    ax.set_title('다층 구조 (Layer Stack)', fontsize=16, fontweight='bold', pad=20)

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
    ax.text(1, 4.3, '2×2 단위 셀', ha='center', fontsize=13,
           fontweight='bold', color='yellow',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.2, n + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Bayer 패턴 배열 (2×2 단위 셀)', fontsize=16, fontweight='bold', pad=20)

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
        (0, 1.5, '#FF6B6B', '실리콘 기판 (Si substrate)'),
        (1.5, 2.0, '#4A90E2', '광다이오드 영역 (Photodiode)\n전하 수집'),
        (3.5, 0.3, '#999999', '금속 배선 (Metal wiring)'),
        (3.8, 0.5, '#95E1D3', '컬러 필터 (Color filter)'),
        (4.3, 0.8, '#87CEEB', '마이크로렌즈 (Microlens)'),
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

    ax.text(x_start + width/2, 7.0, '입사광', ha='center', fontsize=13,
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
    ax.text(x_start + width/2, -0.9, '픽셀 피치 (Pixel pitch) ~1-2μm',
           fontsize=11, ha='center', fontweight='bold')

    ax.set_xlim(0, 11)
    ax.set_ylim(-1.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('이미지 센서 픽셀 구조 (단면도)', fontsize=16, fontweight='bold', pad=20)

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
             linewidth=3, label='입사 파동벡터 $\\mathbf{k}$')

    # Draw projection on xy-plane
    ax.plot([0, kx], [0, ky], [0, 0], 'r--', linewidth=2, alpha=0.5)
    ax.plot([kx, kx], [ky, ky], [0, 1.5+kz], 'r--', linewidth=1, alpha=0.5)

    # Draw theta angle arc (in xz-plane for visualization)
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

    # Draw polarization vectors (perpendicular to k)
    # s-polarization (perpendicular to plane of incidence)
    s_pol = np.array([-np.sin(phi), np.cos(phi), 0])
    s_pol = s_pol / np.linalg.norm(s_pol) * 0.6
    origin_pol = np.array([kx/2, ky/2, 1.5 + kz/2])
    ax.quiver(origin_pol[0], origin_pol[1], origin_pol[2],
             s_pol[0], s_pol[1], s_pol[2],
             color='purple', arrow_length_ratio=0.2, linewidth=2.5,
             label='s-편광 (TE)')

    # p-polarization (in plane of incidence)
    p_pol = np.cross([kx, ky, kz], s_pol)
    p_pol = p_pol / np.linalg.norm(p_pol) * 0.6
    ax.quiver(origin_pol[0], origin_pol[1], origin_pol[2],
             p_pol[0], p_pol[1], p_pol[2],
             color='orange', arrow_length_ratio=0.2, linewidth=2.5,
             label='p-편광 (TM)')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('입사각 및 편광 기하학', fontsize=16, fontweight='bold', pad=20)

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
    ax.text(beam_x - 1.2, 1.5, '입사광\n$\\theta_i$', fontsize=12, color='red', fontweight='bold')

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
    ax.set_title('회절 차수 (Diffraction Orders)', fontsize=16, fontweight='bold', pad=20)

    # Add legend for propagating orders
    ax.text(2.5, 2.5, '전파 차수\n(Propagating)', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/diffraction_orders.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_dir}/diffraction_orders.png")


# Generate all figures
if __name__ == '__main__':
    print("Generating documentation figures...")
    print()

    generate_lattice_structure()
    generate_reciprocal_lattice()
    generate_layer_stack()
    generate_bayer_pattern()
    generate_sensor_structure()
    generate_incidence_geometry()
    generate_diffraction_orders()

    print()
    print(f"All figures generated successfully in {output_dir}/")
