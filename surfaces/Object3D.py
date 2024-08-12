from abc import ABC, abstractmethod
import numpy as np
from surfaces.SurfaceAbs import SurfaceAbs


class Object3D(SurfaceAbs, ABC):
    def __init__(self, position, material_index, index):
        super().__init__(material_index, index)
        self.position = np.array(position)

    def __repr__(self):
        return f"{super().__repr__()}"

    @staticmethod
    def transform_to_camera_coordinates(surface_coords: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
        """
        Helper function to transform 3D surface coordinates to 3D camera coordinates.
        :param surface_coords: Sphere or Cube position in world coordinates.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return: the (x, y, z) coordinates in camera coordinates.
        """
        homogeneous_surface_coords = np.append(surface_coords, 1)

        camera_coords = np.dot(view_matrix, homogeneous_surface_coords)

        return camera_coords[:3] / camera_coords[3]

    @abstractmethod
    def get_enclosing_values(self):
        """
        :return: tuple with values of smallest and biggest x,y,z values of the object
        """
