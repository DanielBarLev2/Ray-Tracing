from util import *
from surfaces.SurfaceAbs import SurfaceAbs


class InfinitePlane(SurfaceAbs):
    def __init__(self, normal, offset, material_index, index):
        super().__init__(material_index, index)
        normal_norm = np.linalg.norm(normal)
        self.normal = np.array(normal) / normal_norm
        self.offset = -offset / normal_norm

    def __repr__(self):
        return f"{super().__repr__()}, normal={self.normal.tolist()}, offset={self.offset}"

    def transform_to_camera(self, view_matrix: np.ndarray):
        """
        Updates the normal vector using the rotation part of the view matrix.
        Then, Updates the new offset by adjusting it with the transformed normal.
         and the translation part of the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return:
        """

        plain_view_matrix = np.linalg.inv(view_matrix).T
        normal = np.dot(plain_view_matrix, np.append(self.normal, self.offset))
        norm = np.linalg.norm(normal[:3])
        self.normal = normal[:3] / norm
        self.offset = normal[3] / norm

    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray | None:
        """
        Computes the intersection between ray and the plane.
        Ray: p_0 + scale * ray_direction
        Plane: normal * X + offset
        Substitute X with ray and solve for scale:= - (normal@ray_source + offset) / normal@ray_direction
        :param ray_source: vector of ray source coordinates.
        :param ray_direction: vector of ray direction coordinates.
        :return: the point in space where intersection between ray and the plane occurs.
        If no intersection is found, None is returned.
        """
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        intersection_angle = np.dot(self.normal, ray_direction)

        # Ray is parallel to the plane
        if np.abs(intersection_angle) <= 1e-12:
            return None

        # Ray lies in the plane
        intersection_point = np.dot(self.normal, ray_source) + self.offset

        scale = - intersection_point / intersection_angle

        # Calculate intersection points where scale is non-negative
        if scale >= 0:
            return ray_source + scale * ray_direction
        else:
            return None

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray):
        """
        Computes the intersection between multiple rays and the plain. using vectorized operations.

        :param rays_sources: N,3 matrix of ray source coordinates
        :param rays_directions: N,3 matrix of ray direction coordinates
        :return: N,3 matrix of points in space where intersection between ray and the plain occurs.
        Entries are np.NaN where no intersection occurs.

        @pre: rays_directions are normalized.
              np.all(np.isclose(np.linalg.norm(rays_directions, axis=-1, keepdims=True), 1.0, atol=EPSILON))
        """
        P0 = rays_sources
        V = rays_directions
        N = self.normal
        d = self.offset

        P0_dot_N = np.dot(P0, N)
        V_dot_N = np.dot(V, N)
        t = (-(P0_dot_N + d) / V_dot_N)

        P = P0 + t[:, np.newaxis] * V

        invalid = (t < 0) | (V_dot_N >= 0) | np.isnan(t)

        P[invalid] = np.nan
        return P

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        return np.full_like(rays_interactions, self.normal)
