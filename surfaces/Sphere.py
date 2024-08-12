from surfaces.Object3D import Object3D
import numpy as np


class Sphere(Object3D):
    def __init__(self, position, radius, material_index, index):
        super().__init__(position, material_index, index)
        self.radius: float = radius

    def __repr__(self):
        position_formatted = [f"{coord:.4f}" for coord in self.position]
        return (f"Sphere(position={position_formatted},"
                f" radius={self.radius:.4f},"
                f" material_index={self.material_index})")

    def transform_to_camera(self, view_matrix: np.ndarray) -> None:
        """
        Transform the center of the sphere using the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        """
        self.position = super().transform_to_camera_coordinates(self.position, view_matrix)

    def intersect(self, ray_source: np.ndarray, ray_direction: np.ndarray) -> np.ndarray | None:
        """
        Computes the intersection between ray and the sphere.
        Ray: p_0 + scale * ray_direction
        Sphere: ||p - O||^2 - r^2
        Substitute p with ray and solve for scale:= ||p_0 + scale * ray_direction - O||^2 - r^2
        => (ray_direction * ray_direction) * scale^2 + 2 * ray_direction * (p_0 - O) * scale + (p_0 - O)^2 - r^2 = 0
        :param ray_source: vector of ray source coordinates
        :param ray_direction: vector of ray direction coordinates
        :return: the point in space where intersection between ray and the sphere occurs.
        If no intersection is found, None is returned.
        """
        ray_direction = ray_direction / (np.linalg.norm(ray_direction))

        p0_minus_O = ray_source - self.position

        # Coefficients of the quadratic equation
        a = np.dot(ray_direction, ray_direction)
        b = 2 * np.dot(ray_direction, p0_minus_O)
        c = np.dot(p0_minus_O, p0_minus_O) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c
        # If the discriminant is negative, ray does not intersect
        if discriminant < 0:
            return None

        sqrt_discriminant = np.sqrt(discriminant)
        scale1 = (-b - sqrt_discriminant) / (2 * a)
        scale2 = (-b + sqrt_discriminant) / (2 * a)

        # If both are negative, ray intersects sphere behind the source
        if 0 <= scale1 and 0 <= scale2:
            return ray_source + min(scale1, scale2) * ray_direction
        elif 0 <= scale1:
            return ray_source + scale1 * ray_direction
        elif 0 <= scale2:
            return ray_source + scale2 * ray_direction

        return None

    def intersect_vectorized(self, rays_sources: np.ndarray, rays_directions: np.ndarray) -> np.ndarray:
        """
        Computes the intersection between multiple rays and the sphere. using vectorized operations.

        :param rays_sources: N,3 matrix of ray source coordinates
        :param rays_directions: N,3 matrix of ray direction coordinates
        :return: N,3 matrix of points in space where intersection between ray and the sphere occurs.
        Entries are np.NaN where no intersection occurs.

        @pre: rays_directions are normalized.
              np.all(np.is close(np.linalg.norm(rays_directions, axis=-1, keep dims=True), 1.0, atol=EPSILON))
        """
        # Calculate coefficients for the quadratic formula
        p0_minus_O = rays_sources - self.position
        a = np.sum(rays_directions * rays_directions, axis=-1)
        b = 2 * np.sum(rays_directions * p0_minus_O, axis=-1)
        c = np.sum(p0_minus_O * p0_minus_O, axis=-1) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c

        intersections = np.full(rays_sources.shape, np.NaN)

        valid_discriminant = 0 <= discriminant

        sqrt_discriminant = np.sqrt(discriminant[valid_discriminant])
        a_valid = a[valid_discriminant]
        b_valid = b[valid_discriminant]
        denominator = 1 / (2 * a_valid)

        scale1 = (-b_valid - sqrt_discriminant) * denominator
        scale2 = (-b_valid + sqrt_discriminant) * denominator

        # Choose the smallest positive scale
        scale_min = np.where(scale1 < scale2, scale1, scale2)
        scale_min = np.where(scale_min < 0, np.maximum(scale1, scale2), scale_min)
        valid_scale = scale_min >= 0

        # Compute the intersection points for valid rays, Ensure non-negative
        valid_discriminant_indices = np.where(valid_discriminant)
        valid_scales_indices = np.where(valid_scale)
        valid_indices = valid_discriminant_indices[0][valid_scales_indices]

        selected_scales = scale_min[valid_scales_indices][:, np.newaxis]
        intersections[valid_indices] = rays_sources[valid_indices] + selected_scales * rays_directions[valid_indices]
        return intersections

    def calculate_normal(self, point: np.ndarray) -> np.ndarray:
        """
         Calculate the normal vector at a given point on the surface. This method computes the normal vector at
         a specific point relative to the surface by normalizing the vector from the surface's position to the point.
        :param point: A numpy array of shape (3, ) representing the coordinates of the point at which the normal vector
         is to be calculated.
        :return: A numpy array of shape (3, ) representing the normal vector at the specified point, normalized to unit
         length.
        """
        direction = (point - self.position)
        return direction / np.linalg.norm(direction)

    def calculate_normals(self, rays_interactions: np.ndarray) -> np.ndarray:
        """
         Calculate the normal vectors at multiple points of interaction with rays on the surface.
         This method computes the normal vectors at each point where rays intersect the surface.
         Each normal vector is calculated by normalizing the vector from the surface's position to each
         intersection point.
        :param rays_interactions: A numpy array of shape (N, 3) where N is the number of intersection points.
         Each row represents the coordinates of an intersection point on the surface.
        :return: A numpy array of shape (N, 3) containing the normal vectors at each intersection point. Each normal
        vector is normalized to unit length.
        """
        directions = (rays_interactions - self.position)
        return directions / np.linalg.norm(directions, axis=-1)[:, np.newaxis]

    def get_enclosing_values(self):
        """
        :return: tuple with values of smallest and biggest x,y,z values of the object
        """
        return (self.position - self.radius), (self.position + self.radius)
