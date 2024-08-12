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
        """
        Calculate the intersection distances of rays with the background. This method provides a default
        implementation where all rays are assumed to intersect the background at the maximum render distance.
        :param rays_sources: A numpy array of shape (N, 3) representing the source points of N rays.
        :param rays_directions: A numpy array of shape (N, 3) representing the direction vectors of N rays.
        :return: A numpy array of shape (N, 3) with intersection distances, each set to the maximum render distance.
        """
        intersections = np.full_like(rays_directions, fill_value=MAX_RENDER_DISTANCE)
        return intersections

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        """
         Calculate the normal vector at a given point on the background. This method returns a zero vector
         as the background has no specific surface normal.
        :param point: A numpy array of shape (3, ) representing the coordinates of the point.
        :return: A numpy array of shape (3, ) representing the normal vector, which is a zero vector for the background.
        """
        return np.zeros_like(point)

    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        """
         Calculate the normal vectors at multiple points of interaction with rays on the background.
         This method returns zero vectors as the background has no specific surface normals.
        :param rays_interactions: A numpy array of shape (N, 3) where N is the number of intersection points.
         Each row represents the coordinates of an intersection point on the background.
        :return: A numpy array of shape (N, 3) containing zero vectors, as the background does not have specific
        surface normals.
        """
        return np.zeros_like(rays_interactions)
