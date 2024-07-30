import numpy as np


class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.material_index = material_index
        self.normal = np.array(normal)
        self.offset = offset

    def transform_to_camera(self, view_matrix: np.ndarray):
        """
        Updates the normal vector using the rotation part of the view matrix.
        Then, Updates the new offset by adjusting it with the transformed normal
         and the translation part of the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return:
        """
        self.normal = np.dot(view_matrix[:3, :3], self.normal)
        self.offset = self.offset - np.dot(self.normal, view_matrix[:3, 3])
