import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_spiral_indices(n):
    indices = []
    left, right = 0, n - 1
    top, bottom = 0, n - 1
    spiral_indices_matrix = np.empty((n, n), dtype=object)


    while left <= right and top <= bottom:
        for i in range(left, right + 1):
            indices.append((top, i))
        top += 1

        for i in range(top, bottom + 1):
            indices.append((i, right))
        right -= 1

        for i in range(right, left - 1, -1):
            indices.append((bottom, i))
        bottom -= 1

        for i in range(bottom, top - 1, -1):
            indices.append((i, left))
        left += 1
    
    i_index = 0
    for i in range(n): 
        for j in range (n):
            spiral_indices_matrix[i,j] = indices[i_index]
            i_index += 1

    return spiral_indices_matrix


def unordered_grid_masked(kx, ky, mask):
    """
    Orders the masked grid points based on row and column indices in ascending order.

    Parameters:
    kx (2D array): The kx grid.
    ky (2D array): The ky grid.
    mask (2D array): A boolean mask indicating which grid points to consider (True for inclusion, False for exclusion).

    Returns:
    list of tuples: The ordered list of grid points where the mask is True.
    2D array: The order of each point.
    """
    # Create a list of all grid points that are masked as True
    grid_points = [(i, j) for i in range(kx.shape[0]) for j in range(kx.shape[1]) if mask[i, j]]
    
    # Sort the grid points by their row (i) and column (j) indices in ascending order
    sorted_grid_points = sorted(grid_points, key=lambda x: (x[0], x[1]))

    # Initialize the order 2D array
    order_2d = np.full(kx.shape, np.nan)

    # Fill the order_2d array based on the sorted grid points
    for idx, (i, j) in enumerate(sorted_grid_points):
        order_2d[i, j] = idx

    return sorted_grid_points, order_2d


def order_grid_points(grid, point):
    """
    Orders the grid points such that all points are adjacent and ordered from far to close to a specified point.

    Parameters:
    grid (2D array): The grid to order.
    point (tuple): The point (x, y) to order the grid points around.

    Returns:
    list of tuples: The ordered list of grid points.
    """
    # Create a list of all grid points
    grid_points = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    
    # Calculate the Euclidean distance of each grid point from the specified point
    distances = np.array([np.sqrt((i - point[0])**2 + (j - point[1])**2) for i, j in grid_points])
    
    # Sort the grid points by their distance from the specified point
    sorted_indices = np.argsort(-distances)  # Sort in descending order (far to close)
    sorted_grid_points = [grid_points[i] for i in sorted_indices]
    
    # Reorder the points to ensure each is adjacent to the previous one
    ordered_grid_points = []
    while sorted_grid_points:
        # Start with the farthest point
        if not ordered_grid_points:
            ordered_grid_points.append(sorted_grid_points.pop(0))
        else:
            # Find the nearest neighbor to the last point in the ordered list
            last_point = ordered_grid_points[-1]
            nearest_idx = np.argmin([np.sqrt((last_point[0] - gp[0])**2 + (last_point[1] - gp[1])**2) for gp in sorted_grid_points])
            ordered_grid_points.append(sorted_grid_points.pop(nearest_idx))
    
    return ordered_grid_points


def order_grid_points_masked(kx, ky, point, mask):
    """
    Orders the masked grid points such that all points are adjacent and ordered from far to close to a specified point.

    Parameters:
    kx (2D array): The kx grid.
    ky (2D array): The ky grid.
    point (tuple): The point (kx, ky) to order the grid points around.
    mask (2D array): A boolean mask indicating which grid points to consider (True for inclusion, False for exclusion).

    Returns:
    list of tuples: The ordered list of grid points where the mask is True.
    2D array: The distances of each point from the specified point.
    """
    # Create a list of all grid points that are masked as True
    grid_points = [(i, j) for i in range(kx.shape[0]) for j in range(kx.shape[1]) if mask[i, j]]
    
    # Calculate the Euclidean distance of each grid point from the specified point
    distances = np.array([np.sqrt((kx[i, j] - point[0])**2 + (ky[i, j] - point[1])**2) for i, j in grid_points])
    
    # Sort the grid points by their distance from the specified point
    sorted_indices = np.argsort(-distances)  # Sort in descending order (far to close)
    sorted_grid_points = [grid_points[i] for i in sorted_indices]

    # Initialize the distances 2D array
    distances_2d = np.full(kx.shape, np.nan)

    # Initialize the distances 2D array
    order_2d = np.full(kx.shape, np.nan)
    
    # Reorder the points to ensure each is adjacent to the previous one
    ordered_grid_points = []
    while sorted_grid_points:
        # Start with the farthest point
        if not ordered_grid_points:
            ordered_grid_points.append(sorted_grid_points.pop(0))
        else:
            # Find the nearest neighbor to the last point in the ordered list
            last_point = ordered_grid_points[-1]
            nearest_idx = np.argmin([np.sqrt((last_point[0] - gp[0])**2 + (last_point[1] - gp[1])**2) for gp in sorted_grid_points])
            ordered_grid_points.append(sorted_grid_points.pop(nearest_idx))
    
    # Fill the distances_2d array based on the order of the grid points
    for idx, (i, j) in enumerate(ordered_grid_points):
        distances_2d[i, j] = idx / (len(ordered_grid_points) - 1)  # Normalize order to range [0, 1]

     # Fill the order_2d array based on the order of the grid points
    for idx, (i, j) in enumerate(ordered_grid_points):
        order_2d[i, j] = idx  # Normalize order to range [0, 1]

    return ordered_grid_points, distances_2d, order_2d


def order_grid_points_start_end(kx, ky, start_point, end_point, mask):
    """
    Orders the masked grid points such that all points are adjacent and ordered from far to close to a specified ending point,
    starting from the grid point closest to the specified starting point.

    Parameters:
    kx (2D array): The kx grid.
    ky (2D array): The ky grid.
    start_point (tuple): The physical space (kx, ky) coordinates of the starting point.
    end_point (tuple): The physical space (kx, ky) coordinates of the ending point.
    mask (2D array): A boolean mask indicating which grid points to consider (True for inclusion, False for exclusion).

    Returns:
    list of tuples: The ordered list of grid points where the mask is True, starting near start_point and ordered toward end_point.
    2D array: The distances of each point from the specified ending point.
    2D array: The order of grid points.
    """
    # Create a list of all grid points that are masked as True
    grid_points = [(i, j) for i in range(kx.shape[0]) for j in range(kx.shape[1]) if mask[i, j]]

    # Find the grid point closest to the start_point
    start_distances = np.array([np.sqrt((kx[i, j] - start_point[0])**2 + (ky[i, j] - start_point[1])**2) for i, j in grid_points])
    start_idx = np.argmin(start_distances)
    start_grid_point = grid_points.pop(start_idx)

    # Calculate the Euclidean distance of each grid point from the specified end_point
    end_distances = np.array([np.sqrt((kx[i, j] - end_point[0])**2 + (ky[i, j] - end_point[1])**2) for i, j in grid_points])
    
    # Sort the grid points by their distance from the specified end_point
    sorted_indices = np.argsort(-end_distances)  # Sort in descending order (far to close)
    sorted_grid_points = [grid_points[i] for i in sorted_indices]

    # Initialize the distances 2D array
    distances_2d = np.full(kx.shape, np.nan)
    order_2d = np.full(kx.shape, np.nan)
    
    # Reorder the points to ensure each is adjacent to the previous one, starting from the nearest to start_point
    ordered_grid_points = [start_grid_point]
    while sorted_grid_points:
        last_point = ordered_grid_points[-1]
        nearest_idx = np.argmin([np.sqrt((last_point[0] - gp[0])**2 + (last_point[1] - gp[1])**2) for gp in sorted_grid_points])
        ordered_grid_points.append(sorted_grid_points.pop(nearest_idx))

    # Fill the distances_2d array and order_2d array based on the order of the grid points
    for idx, (i, j) in enumerate(ordered_grid_points):
        distances_2d[i, j] = idx / (len(ordered_grid_points) - 1)  # Normalize order to range [0, 1]
        order_2d[i, j] = idx

    return ordered_grid_points, distances_2d, order_2d



def plot_ordered_grid_3d(kx, ky, ordered_grid_points, reference_point):
    """
    Plots a 3D surface plot where the height represents the normalized order of the grid points,
    and overlays the reference point as a scatter plot.

    Parameters:
    kx (2D array): The kx grid.
    ky (2D array): The ky grid.
    ordered_grid_points (list of tuples): The ordered list of grid points (i, j).
    reference_point (tuple): The reference point (kx_ref, ky_ref) to be plotted as a scatter plot.
    """
    # Create an empty array to store the order values
    order_array = np.full_like(kx, np.nan, dtype=float)
    
    # Assign normalized order values
    num_points = len(ordered_grid_points)
    for idx, (i, j) in enumerate(ordered_grid_points):
        order_array[i, j] = idx / (num_points - 1)  # Normalize order to range [0, 1]
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kx, ky, order_array, cmap='viridis', edgecolor='none')

    # Plot the reference point as a scatter plot
    kx_ref, ky_ref = reference_point
    ax.scatter(kx_ref, ky_ref, 1, color='red', s=100, label='Reference Point', depthshade=True)

    ax.set_title('3D Plot of Normalized Order of Grid Points')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Normalized Order')
    ax.legend()
    plt.show()


def plot_ordered_grid_histogram(kx, ky, distances_2d):
    """
    Plots a 3D histogram where the height and color represent the normalized order of the grid points.

    Parameters:
    kx (2D array): The kx grid.
    ky (2D array): The ky grid.
    distances_2d (2D array): The 2D array representing the normalized order of grid points.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Flatten the kx, ky, and distances_2d arrays for plotting
    x_flat = kx.flatten()
    y_flat = ky.flatten()
    z_flat = np.zeros_like(x_flat)
    dz_flat = distances_2d.flatten()

    # Filter out NaN values
    mask = ~np.isnan(dz_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]
    dz_flat = dz_flat[mask]

    # Normalize the heights for coloring
    norm = plt.Normalize(dz_flat.min(), dz_flat.max())
    colors = cm.viridis(norm(dz_flat))  # Use the 'viridis' colormap

    # Create the 3D histogram bars with color gradient
    ax.bar3d(x_flat, y_flat, z_flat, 0.1, 0.1, dz_flat, shade=True, color=colors)

    ax.set_title('3D Histogram of Normalized Order of Grid Points')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Normalized Order')

    # Add a color bar
    mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array(dz_flat)
    fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=5)

    plt.show()


def plot_ordered_grid_2d(kx, ky, order_2d):
    """
    Plots a 2D plot where the grid points are labeled with their order.

    Parameters:
    kx (2D array): The kx grid.
    ky (2D array): The ky grid.
    order_2d (2D array): The 2D array representing the order of grid points.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the 2D scatter plot of the grid points
    scatter = ax.scatter(kx, ky, c=order_2d, cmap='viridis', edgecolor='k', s=100)
    
    # Add text annotations to each grid point
    for i in range(order_2d.shape[0]):
        for j in range(order_2d.shape[1]):
            if not np.isnan(order_2d[i, j]):
                ax.text(kx[i, j], ky[i, j], f'{int(order_2d[i, j])}', 
                        color='white', ha='center', va='center', fontsize=3, weight='bold')

    ax.set_title('2D Plot of Grid Points with Order Labels')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    fig.colorbar(scatter, ax=ax, label='Order')

    plt.grid(True)
    plt.show()
