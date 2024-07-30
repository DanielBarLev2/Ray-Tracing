import numpy as np


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up_vector = np.array(up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.look_direction = look_at - position

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
        forward = forward / np.linalg.norm(forward)

        right = np.cross(self.up_vector, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        view_matrix = np.array([[right[0], right[1], right[2], -np.dot(right, self.position)],
                                [up[0], up[1], up[2], -np.dot(up, self.position)],
                                [forward[0], forward[1], forward[2], -np.dot(forward, self.position)],
                                [0, 0, 0, 1]])

        return view_matrix
