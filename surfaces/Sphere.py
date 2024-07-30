import numpy as np
from surfaces.Object3D import Object3D


class Sphere(Object3D):
    def __init__(self, position, radius, material_index):
        super().__init__(material_index)
        self.position = np.array(position)
        self.radius = radius

    def transform_to_camera(self, view_matrix: np.ndarray) -> None:
        """
        Transform the center of the sphere using the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        """
        self.position = super().transform_to_camera_coordinates(self.position, view_matrix)
