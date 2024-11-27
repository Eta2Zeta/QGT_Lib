import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ZoneDivider:
    def __init__(self, kx, ky, num_zones, start_angle=0):
        """
        Initialize the ZoneDivider object.

        Parameters:
        kx (2D array): kx grid.
        ky (2D array): ky grid.
        num_zones (int): Number of zones to divide the grid into.
        start_angle (float): The starting angle for the first zone division in radians.
        """
        self.kx = kx
        self.ky = ky
        self.num_zones = num_zones
        self.start_angle = start_angle
        self.zones = None
        self.zone_boundaries = np.linspace(0, 2 * np.pi, num_zones + 1) + start_angle
        self.zone_boundaries = np.mod(self.zone_boundaries, 2 * np.pi)  # Ensure boundaries are within [0, 2π]

    def calculate_zones(self):
        """
        Calculate the zones for the grid based on angles.
        """
        angles = np.arctan2(self.ky, self.kx)
        angles[angles < 0] += 2 * np.pi
        
        self.zones = np.zeros_like(self.kx, dtype=int)
        for i in range(self.num_zones):
            lower_bound = self.zone_boundaries[i]
            upper_bound = self.zone_boundaries[i + 1]
            if lower_bound < upper_bound:
                self.zones[(angles >= lower_bound) & (angles < upper_bound)] = i
            else:
                # Handles the wrapping case where the angle range crosses 2π
                self.zones[(angles >= lower_bound) | (angles < upper_bound)] = i

    def create_mask_for_zone(self, zone_number):
        """
        Create a mask that is True for grid points in the specified zone and False otherwise.

        Parameters:
        zone_number (int): The zone number to create the mask for.

        Returns:
        mask (2D array): A boolean array where True indicates the point is in the specified zone.
        """
        if self.zones is None:
            raise ValueError("Zones have not been calculated yet. Call calculate_zones() first.")
        
        return self.zones == zone_number


    def get_farthest_points_in_zone(self, zone_number):
        """
        Get the points in the specified zone that are farthest from the origin, with the smallest and largest angles.

        Parameters:
        zone_number (int): The zone number to find the points within.

        Returns:
        tuple: Two tuples, each representing the farthest points with the smallest and largest angles, respectively.
            Each tuple contains the (kx, ky) coordinates.
        """
        mask = self.create_mask_for_zone(zone_number)

        # Find the boundary points
        boundary_mask = np.zeros_like(mask, dtype=bool)
        boundary_mask[:, [0, -1]] = True  # Left and right edges
        boundary_mask[[0, -1], :] = True  # Top and bottom edges

        # Combine the boundary mask with the zone mask
        edge_mask = boundary_mask & mask

        # Calculate the angles for the edge points
        angles = np.arctan2(self.ky, self.kx)
        angles[angles < 0] += 2 * np.pi

        # Filter angles and positions using the edge mask
        edge_angles = angles[edge_mask]
        edge_positions = np.argwhere(edge_mask)

        # Handle wrapping around 2*pi
        if np.ptp(edge_angles) > np.pi:
            edge_angles = np.mod(edge_angles + np.pi, 2 * np.pi)

        # Find the indices of the smallest and largest angles
        smallest_angle_idx = np.argmin(edge_angles)
        largest_angle_idx = np.argmax(edge_angles)

        # Get the (kx, ky) points corresponding to these indices
        smallest_angle_point = (self.kx[tuple(edge_positions[smallest_angle_idx])], 
                                self.ky[tuple(edge_positions[smallest_angle_idx])])
        
        largest_angle_point = (self.kx[tuple(edge_positions[largest_angle_idx])], 
                            self.ky[tuple(edge_positions[largest_angle_idx])])

        return smallest_angle_point, largest_angle_point


    def plot_zones(self):
        """
        Plot the grid zones.
        """
        if self.zones is None:
            raise ValueError("Zones have not been calculated yet. Call calculate_zones() first.")

        plt.figure(figsize=(8, 8))
        plt.contourf(self.kx, self.ky, self.zones, levels=np.arange(self.num_zones + 1) - 0.5, cmap='tab10', alpha=0.7)
        plt.colorbar(ticks=np.arange(self.num_zones), label='Zone Index')
        plt.title(f'Division of 2D Grid into {self.num_zones} Zones')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.grid(True)
        plt.show()

    def plot_zones_3d(self, fig, ax=None, z_value=0):
        """
        Plot the grid zones in 3D as continuous sheets.

        Parameters:
        z_value (float): The constant z-value to be used for 3D plotting.
        """
        if self.zones is None:
            raise ValueError("Zones have not been calculated yet. Call calculate_zones() first.")

        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        # Create a surface for each zone using the mask
        for i in range(self.num_zones):
            mask = self.create_mask_for_zone(i)
            zone_surface = np.full_like(self.kx, z_value, dtype=float)
            
            # Apply the mask to kx, ky, and zone_surface
            masked_kx = np.ma.masked_where(~mask, self.kx)
            masked_ky = np.ma.masked_where(~mask, self.ky)
            masked_surface = np.ma.masked_where(~mask, zone_surface)

            # Plot the surface for the current zone
            ax.plot_surface(masked_kx, masked_ky, masked_surface, label=f'Zone {i+1}', alpha=0.3)

        ax.set_title(f'Division of 2D Grid into {self.num_zones} Zones')
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_zlabel('Zone Index')
        ax.legend()

    def get_zone(self, kx_val, ky_val):
        """
        Get the zone index for a specific (kx, ky) point.

        Parameters:
        kx_val (float): The kx value.
        ky_val (float): The ky value.

        Returns:
        int: The zone index for the specified point.
        """
        angle = np.arctan2(ky_val, kx_val)
        if angle < 0:
            angle += 2 * np.pi
        
        for i in range(self.num_zones):
            lower_bound = self.zone_boundaries[i]
            upper_bound = self.zone_boundaries[i + 1]
            if lower_bound < upper_bound:
                if lower_bound <= angle < upper_bound:
                    return i
            else:
                if angle >= lower_bound or angle < upper_bound:
                    return i
        return None  # In case the angle doesn't fall into any zone, which should not happen
    
    def get_point_with_zone(self, kx_val, ky_val):
        """
        Get a tuple (kx_val, ky_val, zone_number) for a specific (kx, ky) point.

        Parameters:
        kx_val (float): The kx value.
        ky_val (float): The ky value.

        Returns:
        tuple: A tuple (kx_val, ky_val, zone_number).
        """
        zone_number = self.get_zone(kx_val, ky_val)
        return (kx_val, ky_val, zone_number)

    def calculate_distances_in_zone(self, point, zone_number):
        """
        Calculate the distances from each point in the specified zone to the given 2D point.

        Parameters:
        point (tuple): A 2D point (kx_val, ky_val).
        zone_number (int): The zone number to calculate distances within.

        Returns:
        distances (2D array): An array where each element is the distance from the grid point to the given point,
                              with NaN for points outside the specified zone.
        """
        if self.zones is None:
            raise ValueError("Zones have not been calculated yet. Call calculate_zones() first.")
        
        # Calculate the distance from each grid point to the given 2D point
        distances = np.sqrt((self.kx - point[0])**2 + (self.ky - point[1])**2)
        
        # Apply the mask for the zone
        mask = self.create_mask_for_zone(zone_number)
        
        # Set distances outside the specified zone to NaN
        distances[~mask] = np.nan
        
        return distances


def test():
    # Example Usage
    mesh_spacing = 100
    k_max = np.pi

    # Create kx and ky arrays
    kx = np.linspace(-k_max, k_max, mesh_spacing)
    ky = np.linspace(-k_max, k_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)

    fig = plt.figure(figsize=(8,8))

    # Create a ZoneDivider object with 3 zones
    zone_divider = ZoneDivider(kx, ky, num_zones=3)
    zone_divider.calculate_zones()

    # Plot the zones in 3D
    zone_divider.plot_zones_3d(fig, z_value=1)  # Set z_value to any constant for better visualization

    # Calculate the distances in a specific zone to a given point
    kx_val, ky_val = 0.5, 0.5
    zone_number = 0  # For example, zone 0
    distances = zone_divider.calculate_distances_in_zone((kx_val, ky_val), zone_number)

    # Plot the distances in 2D
    plt.show()

