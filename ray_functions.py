from surfaces.SurfaceAbs import SurfaceAbs
from util import *
import numpy as np
from Camera import Camera


def get_ray_vectors(camera: Camera, image_width: int, image_height: int) -> np.ndarray:
    """
    Generates a 3D array of ray vectors for each pixel in the specified image dimensions based on the camera settings.
    :param camera: the camera object with properties defining its screen size and distance.
    :param image_width: the number of horizontal pixels in the image.
    :param image_height: the number of vertical pixels in the image.
    :return: a 3D array (image_height, image_width, 3) where each element is a vector [x, y, z] representing
     the direction of the ray passing through the corresponding pixel.
    """
    w = camera.screen_width
    h = w / image_width * image_height

    h_granularity = h / image_height
    w_granularity = w / image_width

    screen_center = Z_DIRECTION * camera.screen_distance
    screen_pixel_0_0 = screen_center + ((h - h_granularity) / 2 * Y_DIRECTION) - ((w - w_granularity) / 2 * X_DIRECTION)

    i_indices = np.arange(image_height)
    j_indices = np.arange(image_width)
    jj, ii = np.meshgrid(j_indices, i_indices)

    ray_vectors = (
            screen_pixel_0_0
            - (ii[:, :, np.newaxis] * h_granularity * Y_DIRECTION)
            + (jj[:, :, np.newaxis] * w_granularity * X_DIRECTION))

    norms = np.linalg.norm(ray_vectors, axis=2, keepdims=True)
    ray_vectors = ray_vectors / norms

    return ray_vectors


def compute_rays_interactions(surfaces: list[SurfaceAbs],
                              rays_sources: np.ndarray,
                              rays_directions: np.ndarray) -> tuple[list, list]:
    """
    Computes rays interactions with 3D surfaces.
    :param surfaces: list of SurfaceAbs objects representing the 3d surfaces.
    :param rays_sources: matrix of ray source coordinates.
    :param rays_directions: matrix of ray direction coordinates.
    :return: ray_interactions: a list of interactions between all rays and every object in the scene.
             index_list: a list of indices. Signify its object.
    """
    rays_interactions = []
    index_list = []

    for surface in surfaces:
        surface_intersection = surface.intersect_vectorized(rays_sources=rays_sources, rays_directions=rays_directions)
        rays_interactions.append(surface_intersection)
        index_list.append(surface.index)

    return rays_interactions, index_list


def compute_rays_hits(ray_interactions: list[np.ndarray], index_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare distance with the current value in the z-buffer at the corresponding pixel location.
     If the new distance is smaller (the intersection point is closer to the camera),
     update the z-buffer with this new distance.
    :param ray_interactions: a list of interactions between all rays and every object in the scene.
    :param index_list: a list of indices. Signify its surface.
    :return: ray_hits: matrix representing position in space of closest interaction for each ray with any surface.
             surface_indies: matrix representing which surface was interacted by a ray.
    """
    stacked_arrays = np.stack(ray_interactions)
    z_values = stacked_arrays[..., 2]

    # Create a mask to identify None values, replace them with a large number to ignore them
    nan_mask = np.isnan(z_values)
    z_values_with_large_number = np.where(nan_mask, np.inf, z_values)

    min_z_indices = np.argmin(z_values_with_large_number, axis=0)
    i, j = np.ogrid[:stacked_arrays.shape[1], :stacked_arrays.shape[2]]

    ray_hits = stacked_arrays[min_z_indices, i, j]

    indices_array = np.array(index_list)

    surface_indies = indices_array[min_z_indices]

    return ray_hits, surface_indies
