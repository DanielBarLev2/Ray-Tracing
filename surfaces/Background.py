from util import *
from surfaces.SurfaceAbs import SurfaceAbs


class Background(SurfaceAbs):
    def __init__(self):
        super().__init__(material_index=0, index=0)

    def transform_to_camera(self, view_matrix: np.ndarray):
        pass

    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
        pass

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        intersections = np.full_like(rays_directions, fill_value=MAX_RENDER_DISTANCE)
        return intersections

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        return np.zeros_like(point)

    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        return np.zeros_like(rays_interactions)
