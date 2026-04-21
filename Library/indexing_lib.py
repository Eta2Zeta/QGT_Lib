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

