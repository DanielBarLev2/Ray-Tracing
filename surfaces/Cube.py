import numpy as np
from util import *
from surfaces.Object3D import Object3D


class Cube(Object3D):
    def __init__(self, position, scale, material_index, index):
        super().__init__(material_index, index)
        self.position = position
        self.scale = scale
        half_scale = scale/2
        directions = np.array([X_DIRECTION, Y_DIRECTION, Z_DIRECTION])
        face_centers = np.array([position + half_scale * directions,
                                 position - half_scale * directions]).reshape(-1, 3)
        self.right, self.left, self.up, self.down, self.forward, self.backward = face_centers

    def __repr__(self):
        return f"Cube(position={self.position.tolist()}, scale={self.scale}, material_index={self.material_index})"

    def transform_to_camera(self, view_matrix) -> None:
        """
        Transform the center of the cube using the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return:
        """
        self.position = super().transform_to_camera_coordinates(self.position, view_matrix)
        self.right = super().transform_to_camera_coordinates(self.right, view_matrix)
        self.left = super().transform_to_camera_coordinates(self.left, view_matrix)
        self.up = super().transform_to_camera_coordinates(self.up, view_matrix)
        self.down = super().transform_to_camera_coordinates(self.down, view_matrix)
        self.forward = super().transform_to_camera_coordinates(self.forward, view_matrix)
        self.backward = super().transform_to_camera_coordinates(self.backward, view_matrix)

    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray | None:
        """
        Computes the intersection between a ray and the cube's faces.
        Ray: p_0 + t * ray_direction
        Plane: n * (p - p_face) = 0
        Substitute p with the ray equation and solve for t:
        n * ((p_0 + t * ray_direction) - p_face) = 0
        => n * (p_0 - p_face) + t * (n * ray_direction) = 0
        Solve for t: - (n * (p_0 - p_face)) / (n * ray_direction)
        :param ray_source: vector of ray source coordinates
        :param ray_direction: vector of ray direction coordinates
        :return: the point in space where intersection between ray and the cube occurs.
        If no intersection is found, None is returned.
        """
        ray_direction_normalized = ray_direction / np.linalg.norm(ray_direction)
        normals = [self.right - self.left, self.up - self.down, self.forward - self.backward]
        normals = [n / np.linalg.norm(n) for n in normals]

        intersects = []
        for center, normal in zip([self.right, self.left, self.up, self.down, self.forward, self.backward], normals):
            d = np.dot(normal, ray_direction_normalized)
            if np.abs(d) < 1e-12:
                continue

            t = np.dot(normal, center - ray_source) / d
            if t < 0:
                # Intersection behind the ray source
                continue

            intersection = ray_source + t * ray_direction_normalized
            # Check if the intersection point is within the cube face bounds
            if np.all(np.abs(intersection - self.position) <= self.scale / 2):
                intersects.append(intersection)

        if intersects:
            # Return the closest intersection point
            distances = [np.linalg.norm(p - ray_source) for p in intersects]
            return intersects[np.argmin(distances)]

        return None

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        """
        Computes the intersection between multiple rays and the box. using vectorized operations.
        :param rays_sources: matrix of ray source coordinates
        :param rays_directions: matrix of ray direction coordinates
        :return: matrix the of points in space where intersection between ray and the box occurs.
        Entries are None where no intersection occurs.
        """
        ray_directions_normalized = rays_directions / np.linalg.norm(rays_directions, axis=-1, keepdims=True)
        normals = np.array([self.right - self.left, self.up - self.down, self.forward - self.backward])
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize normals

        # Compute plane constants (for each face)
        d_values = np.dot(normals, self.position)
        # Compute the dot products between each ray direction and each normal
        d_normals = np.einsum('ijk,kl->ijl', ray_directions_normalized, normals.T)

        # Avoid division by zero and compute intersections
        avoid_div_zero = 1e-6 < np.abs(d_normals)
        t_values = (d_values - np.einsum('ijk,kl->ijl', rays_sources, normals.T)) / d_normals
        valid_t = (t_values > 0) & avoid_div_zero

        # Calculate intersection points
        intersections = rays_sources[..., np.newaxis, :] + t_values[..., np.newaxis] * ray_directions_normalized[...,
                                                                                       np.newaxis, :]

        # Check bounds for each intersection
        inside_bounds = np.all((intersections >= (self.position - self.scale / 2)) &
                               (intersections <= (self.position + self.scale / 2)), axis=-1)

        # Find valid intersections
        valid_intersections = valid_t & inside_bounds
        # Choose the nearest valid intersection for each ray and face
        nearest_scalar = np.where(valid_intersections, t_values, np.inf).min(axis=-1)
        nearest_intersection = rays_sources + nearest_scalar[..., np.newaxis] * ray_directions_normalized

        # Replace non-intersecting results with NaN or another identifier
        no_intersection = np.isinf(nearest_scalar)
        nearest_intersection[no_intersection] = np.nan

        return nearest_intersection

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate the normal vector for a given point on the surface of the cube.

        :param point: The point on the surface of the cube (a 3D point).
        :return: The normal vector at the given point on the surface of the cube.
        """

        # Calculate the distance from the point to each face center
        dist_right = np.linalg.norm(point - self.right)
        dist_left = np.linalg.norm(point - self.left)
        dist_up = np.linalg.norm(point - self.up)
        dist_down = np.linalg.norm(point - self.down)
        dist_forward = np.linalg.norm(point - self.forward)
        dist_backward = np.linalg.norm(point - self.backward)

        # Find the minimum distance to determine the closest face
        min_dist = min(dist_right, dist_left, dist_up, dist_down, dist_forward, dist_backward)

        # Assign the normal vector based on the closest face
        if np.isclose(min_dist, dist_right):
            normal = (self.right - self.position) / np.linalg.norm(self.right - self.position)
        elif np.isclose(min_dist, dist_left):
            normal = (self.left - self.position) / np.linalg.norm(self.left - self.position)
        elif np.isclose(min_dist, dist_up):
            normal = (self.up - self.position) / np.linalg.norm(self.up - self.position)
        elif np.isclose(min_dist, dist_down):
            normal = (self.down - self.position) / np.linalg.norm(self.down - self.position)
        elif np.isclose(min_dist, dist_forward):
            normal = (self.forward - self.position) / np.linalg.norm(self.forward - self.position)
        elif np.isclose(min_dist, dist_backward):
            normal = (self.backward - self.position) / np.linalg.norm(self.backward - self.position)
        else:
            raise ValueError("The given point is not on the surface of the cube.")

        return normal
