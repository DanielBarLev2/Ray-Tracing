from Ray import Ray
from surfaces.Object3D import Object3D
import numpy as np


class Sphere(Object3D):
    def __init__(self, position, radius, material_index):
        super().__init__(material_index)
        self.position = np.array(position)
        self.radius = radius

    def __repr__(self):
        return f"Sphere(position={self.position.tolist()}, radius={self.radius}, material_index={self.material_index})"


    def transform_to_camera(self, view_matrix: np.ndarray) -> None:
        """
        Transform the center of the sphere using the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        """
        self.position = super().transform_to_camera_coordinates(self.position, view_matrix)

    # todo: check again
    def intersect(self, ray: Ray):
        L = self.position
        p_0 = ray.source
        v = ray.direction/(np.linalg.norm(ray.direction))

        t_ca = np.dot(L, v)
        if t_ca < 0:
            return False, float('inf')

        d_squared = np.linalg.norm(L)**2 - t_ca**2
        if d_squared > self.radius ** 2:
            return False, float('inf')

        t_hc = (self.radius ** 2 - d_squared)**(0.5)
        t = t_ca-t_hc

        p = p_0+t*v

        return True, p
