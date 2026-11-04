# 📡 Phased Array Antenna Visualization Dashboard

A high-performance, interactive visualization dashboard for phased array antenna radiation patterns. This project utilizes the analytical **Dirichlet kernel** for ultra-fast array factor calculations, bypassing traditional brute-force summation. Combined with Matplotlib's optimized update mechanisms, it delivers a smooth, real-time 3D beam-scanning animation.
<div align="center">
  <img src="https://github.com/user-attachments/assets/fbd20a8b-6f4f-40b5-86bb-b8b0cbafdb9b" alt="Phased Array Dashboard Preview" width="850">
  <p><i>Interactive Phased Array Antenna Dashboard</i></p>
</div>
## ✨ Core Features

*   **⚡ Ultra-Fast Analytical Algorithm**: Uses the closed-form Dirichlet kernel to compute the array factor, making calculations hundreds of times faster than traditional methods.
*   **📊 6-in-1 Multi-Dimensional Dashboard**:
    *   **2D Cut Plane**: Real-time tracking of the scan plane, E-plane (phi=0), and H-plane (phi=90).
    *   **3D Radiation Pattern**: Dynamic 3D visualization of the beam shape.
    *   **u-v Space**: Real-time energy distribution in the visible region.
    *   **Theta-Phi Heatmap**: Full-space 2D radiation mapping.
    *   **4 Real-time Cut Monitors**: Bottom panels dynamically display U-axis, V-axis, Diagonal, and Anti-diagonal cuts, eliminating static dead images.
*   **🎬 Smooth Animation Engine**: Utilizes `set_ydata` for 2D updates and object replacement for 3D surfaces, avoiding the flickering and memory leaks caused by `cla()`.
*   **🎨 Modern UI Layout**: Built with `GridSpec` for complex grid layouts. All-English UI prevents font rendering errors and perfectly fits wide screens.

## 🛠️ Requirements

This project requires Python 3.8+ and the following libraries:

```bash
pip install numpy matplotlib
