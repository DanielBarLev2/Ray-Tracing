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

    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
        pass

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        pass

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
