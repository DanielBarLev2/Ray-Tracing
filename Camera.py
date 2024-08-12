from util import *


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up_vector = np.array(up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.look_direction = self.look_at - self.position
        self.z_dir, self.y_dir, self.x_dir = diagonalize_vectors(self.look_direction, self.up_vector)
        self.view_mat = None

    def __repr__(self):
        return f"Camera:\n" \
               f"pos: {self.position}\n" \
               f"look_at: {self.look_at}\n" \
               f"look dir: {self.look_direction}\n" \
               f"up: {self.up_vector}\n" \
               f"screen dist: {self.screen_distance}\n" \
               f"screen width: {self.screen_width}\n"

    def create_view_matrix(self) -> np.ndarray:
        """
        construct a View Matrix using the camera's position, the look-at point, and the up vector.

        Position: The camera's location in world coordinates.
        Look-at Point: Where the camera is pointing. You use this to define the direction the camera is facing.
        Up Vector: Defines which direction is 'up' from the camera's perspective.

        Forward Vector (Z-axis) is normalized vector from the position to the look-at point.
        Right Vector (X-axis) is the cross product of the up vector and the forward vector.
        True Up Vector (Y-axis) is the cross product of the forward vector and the right vector to ensure orthogonality.

        :return: view_matrix that defines the camera as (0, 0, 0) in world coordinates.
        """
        forward = self.look_at - self.position
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
        forward, up, right = diagonalize_vectors(forward, self.up_vector)

        view_matrix = np.array([[right[0], right[1], right[2], -np.dot(right, self.position)],
                                [up[0], up[1], up[2], -np.dot(up, self.position)],
                                [forward[0], forward[1], forward[2], -np.dot(forward, self.position)],
                                [0, 0, 0, 1]])

        self.view_mat = view_matrix
        return view_matrix

    @staticmethod
    def transform_to_camera_coordinates(surface_coords: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
        """
        Helper function to transform 3D surface coordinates to 3D camera coordinates.
        :param surface_coords: Sphere or Cube position in world coordinates.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return: the (x, y, z) coordinates in camera coordinates.
        """
        # Convert surface coordinates to homogeneous coordinates (add a 1 at the end)
        homogeneous_surface_coords = np.append(surface_coords, 1)

        # Transform the coordinates
        camera_coords = np.dot(view_matrix, homogeneous_surface_coords)

        return camera_coords[:3]

    def transform_to_camera(self, view_matrix: np.ndarray):
        self.position = self.transform_to_camera_coordinates(self.position, view_matrix)
        self.look_direction = Z_DIRECTION
        self.up_vector = Y_DIRECTION
