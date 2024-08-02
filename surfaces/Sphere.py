from surfaces.Object3D import Object3D
from util import *
import numpy as np


class Sphere(Object3D):
    def __init__(self, position, radius, material_index, index):
        super().__init__(material_index, index)
        self.position: Vector = np.array(position)
        self.radius: float = radius

    def __repr__(self):
        position_formatted = [f"{coord:.4f}" for coord in self.position]
        return (f"Sphere(position={position_formatted},"
                f" radius={self.radius:.4f},"
                f" material_index={self.material_index})")

    def transform_to_camera(self, view_matrix: np.ndarray) -> None:
        """
        Transform the center of the sphere using the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        """
        self.position = super().transform_to_camera_coordinates(self.position, view_matrix)

    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray | None:
        """
        Computes the intersection between ray and the sphere.
        Ray: p_0 + scale * ray_direction
        Sphere: ||p - O||^2 - r^2
        Substitute p with ray and solve for scale:= ||p_0 + scale * ray_direction - O||^2 - r^2
        => (ray_direction * ray_direction) * scale^2 + 2 * ray_direction * (p_0 - O) * scale + (p_0 - O)^2 - r^2 = 0
        :param ray_source: vector of ray source coordinates
        :param ray_direction: vector of ray direction coordinates
        :return: the point in space where intersection between ray and the sphere occurs.
        If no intersection is found, None is returned.
        """
        ray_direction = ray_direction / (np.linalg.norm(ray_direction))

        p0_minus_O = ray_source - self.position

        # Coefficients of the quadratic equation
        a = np.dot(ray_direction, ray_direction)
        b = 2 * np.dot(ray_direction, p0_minus_O)
        c = np.dot(p0_minus_O, p0_minus_O) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c
        # If the discriminant is negative, ray does not intersect
        if discriminant < 0:
            return None

        sqrt_discriminant = np.sqrt(discriminant)
        scale1 = (-b - sqrt_discriminant) / (2 * a)
        scale2 = (-b + sqrt_discriminant) / (2 * a)

        # If both are negative, ray intersects sphere behind the source
        if 0 <= scale1 and 0 <= scale2:
            return ray_source + min(scale1, scale2) * ray_direction
        elif 0 <= scale1:
            return ray_source + scale1 * ray_direction
        elif 0 <= scale2:
            return ray_source + scale2 * ray_direction

        return None

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        """
        Computes the intersection between multiple rays and the sphere. using vectorized operations.

        :param rays_sources: matrix of ray source coordinates
        :param rays_directions: matrix of ray direction coordinates
        :return: matrix the of points in space where intersection between ray and the sphere occurs.
        Entries are None where no intersection occurs.
        """
        rays_directions = rays_directions / np.linalg.norm(rays_directions, axis=2)[:, :, np.newaxis]

        # Calculate coefficients for the quadratic formula
        p0_minus_O = rays_sources - self.position
        a = np.sum(rays_directions * rays_directions, axis=2)
        b = 2 * np.sum(rays_directions * p0_minus_O, axis=2)
        c = np.sum(p0_minus_O * p0_minus_O, axis=2) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c

        # Initialize intersection points array with NaNs
        intersections = np.full(rays_sources.shape, np.nan)

        # Only proceed where the discriminant is non-negative
        valid = 0 <= discriminant

        sqrt_discriminant = np.sqrt(discriminant[valid])
        a_valid = a[valid]
        b_valid = b[valid]

        scale1 = (-b_valid - sqrt_discriminant) / (2 * a_valid)
        scale2 = (-b_valid + sqrt_discriminant) / (2 * a_valid)

        # Choose the smallest positive scale
        scale_min = np.where(scale1 < scale2, scale1, scale2)
        scale_min = np.where(scale_min < 0, np.maximum(scale1, scale2), scale_min)  # Ensure non-negative
        valid_scale = scale_min >= 0

        # Compute the intersection points for valid rays
        valid_indices = np.where(valid)
        valid_scales_indices = np.where(valid_scale)

        selected_scales = scale_min[valid_scales_indices]
        intersections[valid_indices[0][valid_scales_indices], valid_indices[1][valid_scales_indices], :] = \
            (rays_sources[valid_indices[0][valid_scales_indices], valid_indices[1][valid_scales_indices], :]
             + selected_scales[:, np.newaxis]
             * rays_directions[valid_indices[0][valid_scales_indices], valid_indices[1][valid_scales_indices], :])
        return intersections

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        direction = (point - self.position)
        return direction / np.linalg.norm(direction)

    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        directions = (rays_interactions - self.position)
        return directions / np.linalg.norm(directions, axis=-1)[:, np.newaxis]
