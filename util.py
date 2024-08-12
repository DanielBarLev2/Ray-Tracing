from PIL import Image
import numpy as np
import cv2

ColorVector = np.ndarray
Vector = np.ndarray
Matrix = np.ndarray

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
    """
    Diagonalizes two vectors to create an orthonormal basis. The first vector is assumed to be normalized. The second
    vector is orthogonalized with respect to the first vector, and the third vector is computed as the cross product of
    the first two vectors.

    :param vector1: ndarray of shape (..., 3) representing the first vector, which is assumed to be normalized.
    :param vector2: ndarray of shape (..., 3) representing the second vector to be orthogonalized with respect to vector1.
    :return: Tuple containing:
        - vector1: The input first vector
        - vector2: Orthogonalized and normalized version of the second input vector.
        - vector3: Orthonormal vector computed as the cross product of the normalized vector1 and vector2.

    @pre vector1 is normalized
    """
    vector2_projection = np.sum(vector1 * vector2, axis=-1, keepdims=True)
    vector2 = vector2 - vector2_projection * vector1
    vector2 = vector2 / (np.linalg.norm(vector2, axis=-1, keepdims=True) + EPSILON)

    vector3 = np.cross(vector2, vector1)
    vector3 = vector3 / (np.linalg.norm(vector3, axis=-1, keepdims=True)+EPSILON)

    return vector1, vector2, vector3


def save_image(image_array: np.ndarray, path: str, height: int, width: int) -> None:
    """
    Saves an image represented as a NumPy array to the specified file path after reshaping
    and converting it to an 8-bit unsigned integer format.

    :param image_array: A NumPy array containing the image data. The array should contain values in the range [0, 1].
    :param path: The path where the image will be saved. Should include the file name and extension (e.g., "image.png").
    :param height: The height of the image in pixels.
    :param width: The width of the image in pixels.
    :return: None

    @pre: image_array.size == height * width * 3
    @pre: image_array values are normalized within the range [0, 1]
    @post: The image is saved at the specified path in the format determined by the file extension.
    """

    # Reshape the flattened image array to the specified height, width, and 3 color channels (RGB)
    image_array = image_array.reshape((height, width, 3))

    # Convert the image array from normalized float [0, 1] to 8-bit unsigned integer [0, 255]
    image_array = (image_array * 255).astype(np.uint8)

    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(image_array)

    # Save the image to the specified path
    image.save(path)


def display_image(image_colors: np.ndarray):
    bgr_image = cv2.cvtColor(image_colors, cv2.COLOR_RGB2BGR)
    cv2.imshow('RGB Image', bgr_image)
    cv2.waitKey(0)
