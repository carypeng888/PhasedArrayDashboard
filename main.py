import matplotlib

matplotlib.use('TkAgg')  # 解决 PyCharm 后端冲突

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# 全英文设置，避免字体警告报错
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========================================================================
# 1. Array Basic Parameters
# ========================================================================
c = 3e8
f = 10e9
lam = c / f
k = 2 * np.pi / lam
M, N = 16, 16
dx, dy = lam / 2, lam / 2

theta0, phi0 = 0, 0
u0 = np.sin(np.radians(theta0)) * np.cos(np.radians(phi0))
v0 = np.sin(np.radians(theta0)) * np.sin(np.radians(phi0))


# ========================================================================
# 2. Core Algorithm
# ========================================================================
def dirichlet_kernel(s, N, k, d):
    x_val = k * d * s / 2
    D = np.zeros_like(x_val)
    idx_ok = np.abs(np.sin(x_val)) > 1e-10
    D[idx_ok] = np.sin(N * x_val[idx_ok]) / np.sin(x_val[idx_ok])
    idx_ln = ~idx_ok
    if np.any(idx_ln):
        xt = x_val[idx_ln]
        D[idx_ln] = N * np.cos(N * xt) / (np.cos(xt) + 1e-15)
    return D


def af_fast(U, V, us, vs):
    Dx = dirichlet_kernel(U - us, M, k, dx)
    Dy = dirichlet_kernel(V - vs, N, k, dy)
    return Dx * Dy


# ========================================================================
# 3. Grid Preparation
# ========================================================================
Amax = M * N
th = np.arange(-90, 90.1, 0.5)
th_rad = np.radians(th)

th3 = np.radians(np.arange(0, 92, 2))
ph3 = np.radians(np.arange(0, 362, 2))
PH3, TH3 = np.meshgrid(ph3, th3)
U3 = np.sin(TH3) * np.cos(PH3)
V3 = np.sin(TH3) * np.sin(PH3)
Z3 = np.cos(TH3)

ug = np.linspace(-1, 1, 201)
UG, VG = np.meshgrid(ug, ug)
mask_uv = UG ** 2 + VG ** 2 > 1

thtp = np.arange(0, 91, 1)
phtp = np.arange(0, 361, 2)
PHG, THG = np.meshgrid(np.radians(phtp), np.radians(thtp))
U_tp = np.sin(THG) * np.cos(PHG)
V_tp = np.sin(THG) * np.sin(PHG)

# ========================================================================
# 4. Figure Initialization & Layout
# ========================================================================
fig = plt.figure(figsize=(20, 14), facecolor='#f0f0f0')
fig.canvas.manager.set_window_title('Phased Array Antenna Dashboard')

# Main Grid: 1 row, 2 cols (Left 1.0, Right 2.2)
gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 2.2], wspace=0.15)

# ---------------- Left Column ----------------
gs_left = gs_main[0, 0].subgridspec(2, 1, hspace=0.25)
ax1 = fig.add_subplot(gs_left[0, 0])  # 2D Cut
ax3 = fig.add_subplot(gs_left[1, 0])  # u-v Plane

# ---------------- Right Column ----------------
gs_right = gs_main[0, 1].subgridspec(3, 1, height_ratios=[3.5, 0.8, 1.2], hspace=0.3)
ax2 = fig.add_subplot(gs_right[0, 0], projection='3d')  # 3D Pattern
ax_tp = fig.add_subplot(gs_right[1, 0])  # theta-phi Heatmap

# Bottom 4 panels
gs_bottom = gs_right[2, 0].subgridspec(1, 4, wspace=0.2)
ax4 = fig.add_subplot(gs_bottom[0, 0])
ax5 = fig.add_subplot(gs_bottom[0, 1])
ax6 = fig.add_subplot(gs_bottom[0, 2])
ax7 = fig.add_subplot(gs_bottom[0, 3])

# ==================== Initial Data Plotting ====================
# 1. 2D Cut
F1 = 20 * np.log10(np.abs(
    af_fast(np.sin(th_rad) * np.cos(np.radians(phi0)), np.sin(th_rad) * np.sin(np.radians(phi0)), u0,
            v0)) / Amax + 1e-15)
F2 = 20 * np.log10(np.abs(af_fast(np.sin(th_rad), np.zeros_like(th_rad), u0, v0)) / Amax + 1e-15)
F3 = 20 * np.log10(np.abs(af_fast(np.zeros_like(th_rad), np.sin(th_rad), u0, v0)) / Amax + 1e-15)

h1, = ax1.plot(th, F1, 'b-', linewidth=2.5, label='Scan Plane')
h2, = ax1.plot(th, F2, 'r--', linewidth=2.0, label='phi=0 (x-z)')
h3, = ax1.plot(th, F3, 'g-.', linewidth=2.0, label='phi=90 (y-z)')
h_ln = ax1.plot([theta0, theta0], [-40, 2], 'k:', linewidth=1.5)[0]
ax1.set_xlim([-90, 90])
ax1.set_ylim([-40, 2])
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower left', fontsize=10)
ax1.set_title(f'2D Cut Plane (Pointing: {theta0}°, {phi0}°)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Theta (deg)', fontsize=11)
ax1.set_ylabel('Normalized |E| (dB)', fontsize=11)

# 2. 3D Pattern
dB3 = np.maximum(20 * np.log10(np.abs(af_fast(U3, V3, u0, v0)) / Amax + 1e-15), -40)
r3 = (dB3 + 40) / 40
surf_3d = ax2.plot_surface(r3 * U3, r3 * V3, r3 * Z3, facecolors=plt.cm.jet((dB3 + 40) / 40),
                           rstride=1, cstride=1, linewidth=0, antialiased=False)
line_3d, = ax2.plot([0, u0], [0, v0], [0, np.cos(np.radians(theta0))], 'r-', linewidth=3)

ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-1, 1])
ax2.view_init(elev=35, azim=45)
ax2.set_axis_off()
ax2.set_box_aspect([1, 1, 1], zoom=1.8)
ax2.set_title('3D Radiation Pattern', fontsize=18, fontweight='bold', pad=15)

# 3. u-v Plane
dBg = 20 * np.log10(np.abs(af_fast(UG, VG, u0, v0)) / Amax + 1e-15)
dBg[mask_uv] = np.nan
im_uv = ax3.imshow(dBg, extent=[-1, 1, -1, 1], origin='lower', cmap='jet', vmin=-40, vmax=0, aspect='equal')
ax3.plot(np.cos(np.linspace(0, 2 * np.pi, 181)), np.sin(np.linspace(0, 2 * np.pi, 181)), 'w-', linewidth=1.5)
h_pt_uv, = ax3.plot(u0, v0, 'r+', markersize=16, markeredgewidth=2)
ax3.set_title('u-v Plane (Real-time)', fontsize=16, fontweight='bold')
ax3.set_xlabel('u', fontsize=13)
ax3.set_ylabel('v', fontsize=13)

# 4. theta-phi Heatmap
dBtp = 20 * np.log10(np.abs(af_fast(U_tp, V_tp, u0, v0)) / Amax + 1e-15)
im_tp = ax_tp.imshow(dBtp, extent=[0, 360, 0, 90], origin='lower', cmap='jet', vmin=-40, vmax=0, aspect='auto')
h_pt_tp, = ax_tp.plot(phi0, theta0, 'r+', markersize=12, markeredgewidth=2)
ax_tp.set_title('Theta-Phi Heatmap', fontsize=14, fontweight='bold')
ax_tp.set_xlabel('Phi (deg)', fontsize=11)
ax_tp.set_ylabel('Theta (deg)', fontsize=11)

# 5. Four Dynamic Cut Panels (Replacing static images)
# Generate cut coordinates
u_cut = np.linspace(-1, 1, 201)
v_zero = np.zeros_like(u_cut)
u_zero = np.zeros_like(u_cut)
diag = np.linspace(-1 / np.sqrt(2), 1 / np.sqrt(2), 201)

axes_bottom = [ax4, ax5, ax6, ax7]
titles = ['U-Axis Cut (V=0)', 'V-Axis Cut (U=0)', 'Diagonal (V=U)', 'Anti-Diagonal (V=-U)']
lines_bottom = []

for i, ax in enumerate(axes_bottom):
    line, = ax.plot(u_cut, np.zeros_like(u_cut), 'b-', linewidth=2)
    lines_bottom.append(line)
    ax.set_ylim(-40, 2)
    ax.set_title(titles[i], fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('Spatial Coordinate', fontsize=9)
    ax.set_ylabel('dB', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)

# ========================================================================
# 5. Dynamic Animation Loop
# ========================================================================
plt.draw()
plt.pause(0.1)
print("Animation started... (Close window to stop)")

try:
    for theta_scan in range(0, 61, 2):
        for phi_scan in range(0, 361, 15):
            if not plt.fignum_exists(fig.number):
                raise KeyboardInterrupt

            us = np.sin(np.radians(theta_scan)) * np.cos(np.radians(phi_scan))
            vs = np.sin(np.radians(theta_scan)) * np.sin(np.radians(phi_scan))

            # 1. Update 2D Cut
            F1_new = 20 * np.log10(np.abs(
                af_fast(np.sin(th_rad) * np.cos(np.radians(phi_scan)), np.sin(th_rad) * np.sin(np.radians(phi_scan)),
                        us, vs)) / Amax + 1e-15)
            F2_new = 20 * np.log10(np.abs(af_fast(np.sin(th_rad), np.zeros_like(th_rad), us, vs)) / Amax + 1e-15)
            F3_new = 20 * np.log10(np.abs(af_fast(np.zeros_like(th_rad), np.sin(th_rad), us, vs)) / Amax + 1e-15)

            h1.set_ydata(F1_new)
            h2.set_ydata(F2_new)
            h3.set_ydata(F3_new)
            h_ln.set_xdata([theta_scan, theta_scan])
            ax1.set_title(f'2D Cut Plane (Pointing: {theta_scan}°, {phi_scan}°)', fontsize=14, fontweight='bold')

            # 2. Update 3D Pattern
            surf_3d.remove()
            dB3_new = np.maximum(20 * np.log10(np.abs(af_fast(U3, V3, us, vs)) / Amax + 1e-15), -40)
            r3_new = (dB3_new + 40) / 40
            surf_3d = ax2.plot_surface(r3_new * U3, r3_new * V3, r3_new * Z3,
                                       facecolors=plt.cm.jet((dB3_new + 40) / 40),
                                       rstride=1, cstride=1, linewidth=0, antialiased=False)
            line_3d.set_data([0, us], [0, vs])
            line_3d.set_3d_properties([0, np.cos(np.radians(theta_scan))])

            # 3. Update u-v Plane
            dBg_new = 20 * np.log10(np.abs(af_fast(UG, VG, us, vs)) / Amax + 1e-15)
            dBg_new[mask_uv] = np.nan
            im_uv.set_data(dBg_new)
            h_pt_uv.set_data([us], [vs])

            # 4. Update Heatmap
            dBtp_new = 20 * np.log10(np.abs(af_fast(U_tp, V_tp, us, vs)) / Amax + 1e-15)
            im_tp.set_data(dBtp_new)
            h_pt_tp.set_data([phi_scan], [theta_scan])

            # 5. Update 4 Bottom Cut Panels (The Magic!)
            # U cut
            lines_bottom[0].set_ydata(20 * np.log10(np.abs(af_fast(u_cut, v_zero, us, vs)) / Amax + 1e-15))
            # V cut
            lines_bottom[1].set_ydata(20 * np.log10(np.abs(af_fast(u_zero, u_cut, us, vs)) / Amax + 1e-15))
            # Diagonal
            lines_bottom[2].set_ydata(20 * np.log10(np.abs(af_fast(diag, diag, us, vs)) / Amax + 1e-15))
            # Anti-Diagonal
            lines_bottom[3].set_ydata(20 * np.log10(np.abs(af_fast(diag, -diag, us, vs)) / Amax + 1e-15))

            # Refresh Canvas
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.01)

except KeyboardInterrupt:
    print("Animation stopped manually.")

plt.show()