import numpy as np

ColorVector = np.ndarray
Vector = np.ndarray
Matrix = np.ndarray
Image = np.ndarray

X_DIRECTION = np.array([1, 0, 0])
Y_DIRECTION = np.array([0, 1, 0])
Z_DIRECTION = np.array([0, 0, 1])

EPSILON = 1e-6

MAX_RENDER_DISTANCE = 100_000_000


def normalize_to_color(matrix):
    """
    Normalize a 3D matrix of vectors so that all values are between 0 and 255,
    handling NaN values by replacing them with the minimum value in the matrix.
    Lower values will be mapped to brighter colors.

    :param matrix: A 3D numpy array of shape (h, w, d).
    :return: A 3D numpy array of the same shape with values between 0 and 255.
    """
    # Create a mask for NaN values
    nan_mask = np.isnan(matrix)

    # Replace NaN values with the minimum value of the matrix (ignoring NaNs)
    min_val = np.nanmin(matrix)
    max_val = np.nanmax(matrix)

    # Copy the matrix and replace NaNs with the minimum value
    normalized_matrix = np.copy(matrix)
    normalized_matrix[nan_mask] = min_val

    # Normalize the matrix to the range [0, 1]
    normalized_matrix = (normalized_matrix - min_val) / (max_val - min_val)

    # Invert the normalized values
    inverted_matrix = 1 - normalized_matrix

    # Scale to the range [0, 255]
    scaled_matrix = (inverted_matrix * 255).astype(np.uint8)

    # Optionally, set NaN locations back to a specific value (e.g., 255 for bright color)
    scaled_matrix[nan_mask] = 255  # or any other value to represent NaNs

    return scaled_matrix


def diagonalize_vectors(vector1: Vector, vector2: Vector):
    # normalize the first vector
    vector1 = vector1 / np.linalg.norm(vector1)

    vector2_projection = np.sum(vector1 * vector2, axis=-1, keepdims=True)
    vector2 = vector2 - vector2_projection * vector1
    vector2 = vector2 / (np.linalg.norm(vector2, axis=-1, keepdims=True) + EPSILON)

    vector3 = np.cross(vector2, vector1)
    vector3 = vector3 / np.linalg.norm(vector3)

    return vector1, vector2, vector3
