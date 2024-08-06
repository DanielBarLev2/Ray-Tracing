from util import *
from surfaces.Object3D import Object3D


class Cube(Object3D):
    def __init__(self, position, scale, material_index, index):
        super().__init__(material_index, index)
        self.position = position
        self.scale = scale
        half_scale = scale/2
        directions = np.array([X_DIRECTION, Y_DIRECTION, Z_DIRECTION])
        face_centers = np.array([position + half_scale * directions,
                                 position - half_scale * directions]).reshape(-1, 3)
        self.right, self.left, self.up, self.down, self.forward, self.backward = face_centers

    def __repr__(self):
        return f"Cube(position={self.position.tolist()}, scale={self.scale}, material_index={self.material_index})"

    def transform_to_camera(self, view_matrix) -> None:
        """
        Transform the center of the cube using the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return:
        """
        self.position = super().transform_to_camera_coordinates(self.position, view_matrix)
        self.right = super().transform_to_camera_coordinates(self.right, view_matrix)
        self.left = super().transform_to_camera_coordinates(self.left, view_matrix)
        self.up = super().transform_to_camera_coordinates(self.up, view_matrix)
        self.down = super().transform_to_camera_coordinates(self.down, view_matrix)
        self.forward = super().transform_to_camera_coordinates(self.forward, view_matrix)
        self.backward = super().transform_to_camera_coordinates(self.backward, view_matrix)

    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray | None:
        """
        Computes the intersection between a ray and the cube's faces.
        Ray: p_0 + t * ray_direction
        Plane: n * (p - p_face) = 0
        Substitute p with the ray equation and solve for t:
        n * ((p_0 + t * ray_direction) - p_face) = 0
        => n * (p_0 - p_face) + t * (n * ray_direction) = 0
        Solve for t: - (n * (p_0 - p_face)) / (n * ray_direction)
        :param ray_source: vector of ray source coordinates
        :param ray_direction: vector of ray direction coordinates
        :return: the point in space where intersection between ray and the cube occurs.
        If no intersection is found, None is returned.
        """
        ray_direction_normalized = ray_direction / np.linalg.norm(ray_direction)
        normals = [self.right - self.left, self.up - self.down, self.forward - self.backward]
        normals = [n / np.linalg.norm(n) for n in normals]

        intersects = []
        for center, normal in zip([self.right, self.left, self.up, self.down, self.forward, self.backward], normals):
            d = np.dot(normal, ray_direction_normalized)
            if np.abs(d) < 1e-12:
                continue

            t = np.dot(normal, center - ray_source) / d
            if t < 0:
                # Intersection behind the ray source
                continue

            intersection = ray_source + t * ray_direction_normalized
            # Check if the intersection point is within the cube face bounds
            if np.all(np.abs(intersection - self.position) <= self.scale / 2):
                intersects.append(intersection)

        if intersects:
            # Return the closest intersection point
            distances = [np.linalg.norm(p - ray_source) for p in intersects]
            return intersects[np.argmin(distances)]

        return None

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        """
        Computes the intersection between multiple rays and the box using vectorized operations.
        :param rays_sources: Nx3 matrix of ray source coordinates
        :param rays_directions: Nx3 matrix of ray direction coordinates
        :return: Nx3 matrix of points in space where intersections between rays and the box occur.
        Entries are None where no intersection occurs.
        """
        # Initialize parameters
        inv_dir = 1.0 / rays_directions
        bounds_min = self.position - self.scale / 2
        bounds_max = self.position + self.scale / 2

        # Calculate intersection times
        t_min = (bounds_min - rays_sources) * inv_dir
        t_max = (bounds_max - rays_sources) * inv_dir

        # Reorder t_min and t_max
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        # Calculate the maximum t_min and minimum t_max
        t_near = np.max(t1, axis=2)
        t_far = np.min(t2, axis=2)

        # Check for valid intersections
        valid_intersections = t_near < t_far
        valid_intersections &= t_near > 0

        # Calculate intersection points
        intersections = rays_sources + t_near[..., None] * rays_directions

        # Mask invalid intersections
        intersections[~valid_intersections] = np.nan

        return intersections

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate the normal vector for a given point on the surface of the cube.

        :param point: The point on the surface of the cube (a 3D point).
        :return: The normal vector at the given point on the surface of the cube.
        """

        # Calculate the distance from the point to each face center
        dist_right = np.linalg.norm(point - self.right)
        dist_left = np.linalg.norm(point - self.left)
        dist_up = np.linalg.norm(point - self.up)
        dist_down = np.linalg.norm(point - self.down)
        dist_forward = np.linalg.norm(point - self.forward)
        dist_backward = np.linalg.norm(point - self.backward)

        # Find the minimum distance to determine the closest face
        min_dist = min(dist_right, dist_left, dist_up, dist_down, dist_forward, dist_backward)

        # Assign the normal vector based on the closest face
        if np.isclose(min_dist, dist_right):
            normal = (self.right - self.position) / np.linalg.norm(self.right - self.position)
        elif np.isclose(min_dist, dist_left):
            normal = (self.left - self.position) / np.linalg.norm(self.left - self.position)
        elif np.isclose(min_dist, dist_up):
            normal = (self.up - self.position) / np.linalg.norm(self.up - self.position)
        elif np.isclose(min_dist, dist_down):
            normal = (self.down - self.position) / np.linalg.norm(self.down - self.position)
        elif np.isclose(min_dist, dist_forward):
            normal = (self.forward - self.position) / np.linalg.norm(self.forward - self.position)
        elif np.isclose(min_dist, dist_backward):
            normal = (self.backward - self.position) / np.linalg.norm(self.backward - self.position)
        else:
            raise ValueError("The given point is not on the surface of the cube.")

        return normal

    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        """
        Calculate the normal vectors for given points on the surface of the cube.

        :param rays_interactions: A ndarray of points on the surface of the cube (shape: Nx3).
        :return: A ndarray of normal vectors at the given points on the surface of the cube (shape: Nx3).
        """
        # Calculate the distance from the points to each face center
        dist_right = np.linalg.norm(rays_interactions - self.right, axis=1)
        dist_left = np.linalg.norm(rays_interactions - self.left, axis=1)
        dist_up = np.linalg.norm(rays_interactions - self.up, axis=1)
        dist_down = np.linalg.norm(rays_interactions - self.down, axis=1)
        dist_forward = np.linalg.norm(rays_interactions - self.forward, axis=1)
        dist_backward = np.linalg.norm(rays_interactions - self.backward, axis=1)

        # Stack the distances to find the minimum distance
        dists = np.stack([dist_right, dist_left, dist_up, dist_down, dist_forward, dist_backward], axis=1)
        min_indices = np.argmin(dists, axis=1)

        # Create an array to hold the normal vectors
        normals = np.zeros(rays_interactions.shape)

        # Assign the normal vectors based on the closest face
        normals[min_indices == 0] = (self.right - self.position) / np.linalg.norm(self.right - self.position)
        normals[min_indices == 1] = (self.left - self.position) / np.linalg.norm(self.left - self.position)
        normals[min_indices == 2] = (self.up - self.position) / np.linalg.norm(self.up - self.position)
        normals[min_indices == 3] = (self.down - self.position) / np.linalg.norm(self.down - self.position)
        normals[min_indices == 4] = (self.forward - self.position) / np.linalg.norm(self.forward - self.position)
        normals[min_indices == 5] = (self.backward - self.position) / np.linalg.norm(self.backward - self.position)

        return normals
