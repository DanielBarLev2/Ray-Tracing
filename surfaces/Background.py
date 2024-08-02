import numpy as np

from SurfaceAbs import SurfaceAbs


class Background(SurfaceAbs):
    def __init__(self, color):
        super().__init__()
        self.color = color

    def transform_to_camera(self, view_matrix: np.ndarray):
        pass

    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
        pass

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        pass

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        pass

    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        pass
