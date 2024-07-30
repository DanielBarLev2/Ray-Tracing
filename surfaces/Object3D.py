from abc import ABC, abstractmethod
import numpy as np


class Object3D(ABC):
    def __init__(self, material_index):
        self.material_index = material_index

    @staticmethod
    def transform_to_camera_coordinates(surface_coords: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
        """
        Helper function to transform a 3D surface coordinates to a 3D camera coordinates.
        :param surface_coords: Sphere or Cube position in world coordinates.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return: the (x, y, z) coordinates of the camera coordinates.
        """
        # Convert surface coordinates to homogeneous coordinates (add a 1 at the end)
        homogeneous_surface_coords = np.append(surface_coords, 1)

        camera_coords = np.dot(view_matrix, homogeneous_surface_coords)

        return camera_coords[:3]

    @abstractmethod
    def transform_to_camera(self, view_matrix: np.ndarray):
        """This method must be overridden in subclasses"""
        pass
