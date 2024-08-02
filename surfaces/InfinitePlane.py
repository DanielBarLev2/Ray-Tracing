from util import *
from surfaces.SurfaceAbs import SurfaceAbs


class InfinitePlane(SurfaceAbs):
    def __init__(self, normal, offset, material_index, index):
        super().__init__(material_index, index)
        self.normal = np.array(normal) / np.linalg.norm(normal)
        self.offset = offset

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
        self.normal = np.dot(view_matrix[:3, :3], self.normal)
        self.normal = np.array(self.normal) / np.linalg.norm(self.normal)  # normalize
        self.offset = -self.offset - np.dot(self.normal, view_matrix[:3, 3])

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

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        """
        Computes the intersection between multiple rays and the plane. using vectorized operations.

        :param rays_sources: matrix of ray source coordinates.
        :param rays_directions: matrix of ray direction coordinates.
        :return: matrix the of points in space where intersection between ray and the plane occurs.
        Entries are None where no intersection occurs.
        """
        rays_directions = rays_directions / np.linalg.norm(rays_directions, axis=2)[:, :, np.newaxis]

        intersection_angles = np.dot(rays_directions, self.normal)

        intersection_points = np.dot(rays_sources, self.normal) + self.offset
        valid_intersection_position = 1e-12 <= np.abs(intersection_angles)

        intersections = np.full_like(rays_sources, np.nan)

        # Where valid, compute the scale
        scale = np.where(valid_intersection_position, - (intersection_points / intersection_angles), np.nan)

        # Calculate intersection points where scale is non-negative
        valid_intersection = valid_intersection_position & (0 <= scale)
        intersections[valid_intersection] = (rays_sources[valid_intersection]
                                             + scale[valid_intersection][:, np.newaxis]
                                             * rays_directions[valid_intersection])
        return intersections

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        return np.full_like(rays_interactions, self.normal)
