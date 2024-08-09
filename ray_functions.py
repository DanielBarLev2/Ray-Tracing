from surfaces.SurfaceAbs import SurfaceAbs
from util import *
import numpy as np
from Camera import Camera


def get_ray_vectors(camera: Camera, rays_sources: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
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

    screen_center = camera.position + camera.z_dir * camera.screen_distance
    screen_pixel_0_0 = screen_center + ((h - h_granularity) / 2 * camera.y_dir) - (
                (w - w_granularity) / 2 * camera.x_dir)

    i_indices = np.arange(image_height)
    j_indices = np.arange(image_width)
    jj, ii = np.meshgrid(j_indices, i_indices)

    ray_destinations = (
            screen_pixel_0_0
            - (ii[:, :, np.newaxis] * h_granularity * camera.y_dir)
            + (jj[:, :, np.newaxis] * w_granularity * camera.x_dir))

    ray_vectors = ray_destinations - rays_sources
    norms = np.linalg.norm(ray_vectors, axis=2, keepdims=True)
    ray_vectors = ray_vectors / norms
    ray_vectors = ray_vectors.reshape(-1, ray_vectors.shape[-1])

    return ray_vectors


def compute_rays_interactions(surfaces: list[SurfaceAbs],
                              rays_sources: np.ndarray,
                              rays_directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes rays interactions with 3D surfaces.
    :param surfaces: list of SurfaceAbs objects representing the 3d surfaces.
    :param rays_sources: matrix of ray source coordinates.
    :param rays_directions: matrix of ray direction coordinates.
    :return: ray_interactions: a list of interactions between all rays and every object in the scene.
             index_list: a list of indices. Signify its object.
    """
    closest_intersections = np.full_like(rays_directions, np.nan)
    closest_intersections_dist = np.full(rays_directions.shape[0], np.inf)
    closest_intersections_indices = np.zeros(rays_directions.shape[0], dtype=int)
    for surface in surfaces:
        surface_intersection = surface.intersect_vectorized(rays_sources=rays_sources, rays_directions=rays_directions)
        invalid_mask = np.all(np.isnan(surface_intersection), axis=-1)
        if np.all(invalid_mask):
            continue

        surface_intersection_dist = np.linalg.norm(surface_intersection - rays_sources, axis=-1)

        smaller_mask = surface_intersection_dist < closest_intersections_dist
        update_mask = smaller_mask & (~invalid_mask)

        closest_intersections[update_mask] = surface_intersection[update_mask]
        closest_intersections_dist[update_mask] = surface_intersection_dist[update_mask]
        closest_intersections_indices[update_mask] = surface.index

    return closest_intersections, closest_intersections_indices


def compute_rays_hits(ray_sources: np.ndarray, ray_interactions: list[np.ndarray], index_list: list[int]) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Compare the distance from the ray source to each interaction point in space.
    If the new distance is smaller (the interaction point is closer to the ray origin),
    update to reflect this closest interaction.

    :param ray_sources: an ndarray of the same size as ray_interactions, representing the sources of the rays.
    :param ray_interactions: a list of interactions between all rays and every object in the scene.
    :param index_list: a list of indices representing the surface each ray interacts with.
    :return: ray_hits: matrix representing the position in space of the closest interaction for each ray.
             surface_indices: matrix representing which surface was interacted by a ray.
    """
    # Stack the ray_interactions list into a single numpy array
    stacked_arrays = np.stack(ray_interactions)  # Shape: (num_interactions, h*w, c)

    # Calculate distances from the ray source to each interaction point
    distances = np.linalg.norm(stacked_arrays - ray_sources, axis=-1)  # Shape: (num_interactions, h*w)

    # Create a mask to identify NaN values, replace them with a large number to ignore them
    nan_mask = np.isnan(distances)
    distances_with_large_number = np.where(nan_mask, np.inf, distances)  # Shape: (num_interactions, h*w)

    # Find the indices of the minimum distances
    min_dist_indices = np.argmin(distances_with_large_number, axis=0)  # Shape: (h*w)

    # Use these indices to gather the closest interaction points
    ray_hits = stacked_arrays[min_dist_indices, np.arange(stacked_arrays.shape[1])]  # Shape: (h*w, c)

    # Convert the index_list to a numpy array
    indices_array = np.array(index_list)  # Shape: (num_interactions,)

    # Get the surface indices using the min_dist_indices
    surface_indices = indices_array[min_dist_indices]  # Shape: (h*w)

    return ray_hits, surface_indices


def get_closest_hits(rays_sources: Matrix, rays_directions: Matrix, surfaces: list[SurfaceAbs]) \
        -> tuple[Matrix, Matrix]:
    rays_interactions, index_list = compute_rays_interactions(surfaces, rays_sources, rays_directions)
    return rays_interactions, index_list
    if len(rays_interactions) == 0:
        return np.zeros_like(rays_sources), np.zeros_like(rays_sources)
    print(f"hits {len(rays_interactions)}")
    res = compute_rays_hits(ray_sources=rays_sources, ray_interactions=rays_interactions, index_list=index_list)
    ray_hits, surface_indices = res
    return ray_hits, surface_indices
