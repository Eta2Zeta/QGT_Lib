import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Define test functions
def fx(kx, ky, zeta):
    return np.sin(zeta * (np.cos(kx) + np.cos(ky)))

def fy(kx, ky, zeta):
    return np.cos(zeta * (np.cos(kx) + np.cos(ky)))

def numerical_derivative(func, kx, ky, zeta, dx=1e-5, dy=1e-5, order=3):
    df_dx = derivative(lambda kx_: func(kx_, ky, zeta), kx, dx=dx, order=order)
    df_dy = derivative(lambda ky_: func(kx, ky_, zeta), ky, dx=dy, order=order)
    d2f_dxx = derivative(lambda kx_: derivative(lambda kx__: func(kx__, ky, zeta), kx_, dx=dx, order=order), kx, dx=dx, order=order)
    d2f_dyy = derivative(lambda ky_: derivative(lambda ky__: func(kx, ky__, zeta), ky_, dx=dy, order=order), ky, dx=dy, order=order)
    d2f_dxy = derivative(lambda kx_: derivative(lambda ky_: func(kx_, ky_, zeta), ky, dx=dy, order=order), kx, dx=dx, order=order)
    return df_dx, df_dy, d2f_dxx, d2f_dyy, d2f_dxy

def plot_derivatives(func, analytical_derivatives, zeta, title_prefix):
    kx = np.linspace(-np.pi, np.pi, 100)
    ky = np.linspace(-np.pi, np.pi, 100)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    deriv_names = ["df/dx", "df/dy", "d2f/dxx", "d2f/dyy", "d2f/dxy"]
    
    numerical_values_list = [np.zeros_like(kx_grid) for _ in range(5)]
    analytical_values_list = [np.zeros_like(kx_grid) for _ in range(5)]
    
    for ix in range(kx_grid.shape[0]):
        for iy in range(kx_grid.shape[1]):
            numerical_values = numerical_derivative(func, kx_grid[ix, iy], ky_grid[ix, iy], zeta)
            for i in range(5):
                numerical_values_list[i][ix, iy] = numerical_values[i]
                analytical_values_list[i][ix, iy] = analytical_derivatives[i](kx_grid[ix, iy], ky_grid[ix, iy])
    
    for i, deriv_name in enumerate(deriv_names):
        ax_numerical = axes[0, i]
        cs_numerical = ax_numerical.contourf(kx_grid, ky_grid, numerical_values_list[i], cmap='coolwarm')
        fig.colorbar(cs_numerical, ax=ax_numerical)
        ax_numerical.set_title(f"{title_prefix} {deriv_name} (Numerical)")
        
        ax_analytical = axes[1, i]
        cs_analytical = ax_analytical.contourf(kx_grid, ky_grid, analytical_values_list[i], cmap='coolwarm')
        fig.colorbar(cs_analytical, ax=ax_analytical)
        ax_analytical.set_title(f"{title_prefix} {deriv_name} (Analytical)")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f"{title_prefix}_derivatives.png")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_all_derivatives(zeta):
    analytical_fx = [
        lambda kx, ky: -zeta * np.sin(kx) * np.cos(zeta * (np.cos(kx) + np.cos(ky))),
        lambda kx, ky: -zeta * np.sin(ky) * np.cos(zeta * (np.cos(kx) + np.cos(ky))),
        lambda kx, ky: -zeta * (np.cos(kx) * np.cos(zeta * (np.cos(kx) + np.cos(ky))) + zeta * np.sin(kx)**2 * np.sin(zeta * (np.cos(kx) + np.cos(ky)))),
        lambda kx, ky: -zeta * (np.cos(ky) * np.cos(zeta * (np.cos(kx) + np.cos(ky))) + zeta * np.sin(ky)**2 * np.sin(zeta * (np.cos(kx) + np.cos(ky)))),
        lambda kx, ky: -zeta**2 * np.sin(kx) * np.sin(ky) * np.sin(zeta * (np.cos(kx) + np.cos(ky)))
    ]
    
    analytical_fy = [
        lambda kx, ky: zeta * np.sin(kx) * np.sin(zeta * (np.cos(kx) + np.cos(ky))),
        lambda kx, ky: zeta * np.sin(ky) * np.sin(zeta * (np.cos(kx) + np.cos(ky))),
        lambda kx, ky: zeta * (np.cos(kx) * np.sin(zeta * (np.cos(kx) + np.cos(ky))) - zeta * np.sin(kx)**2 * np.cos(zeta * (np.cos(kx) + np.cos(ky)))),
        lambda kx, ky: zeta * (np.cos(ky) * np.sin(zeta * (np.cos(kx) + np.cos(ky))) - zeta * np.sin(ky)**2 * np.cos(zeta * (np.cos(kx) + np.cos(ky)))),
        lambda kx, ky: -zeta**2 * np.sin(kx) * np.sin(ky) * np.cos(zeta * (np.cos(kx) + np.cos(ky)))
    ]
    
    plot_derivatives(fx, analytical_fx, zeta, "fx")
    plot_derivatives(fy, analytical_fy, zeta, "fy")

zeta = 1
test_all_derivatives(zeta)
