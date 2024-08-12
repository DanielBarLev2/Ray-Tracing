from surfaces.SurfaceAbs import SurfaceAbs
from util import *
import numpy as np
from Camera import Camera


def get_initial_rays(camera: Camera, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
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

    rays_destinations = (
            screen_pixel_0_0
            - (ii[:, :, np.newaxis] * h_granularity * camera.y_dir)
            + (jj[:, :, np.newaxis] * w_granularity * camera.x_dir))

    rays_destinations = rays_destinations.reshape((image_height * image_width, 3))
    rays_sources = np.full((image_height * image_width, 3), camera.position)

    rays_directions = rays_destinations - rays_sources
    norms = np.linalg.norm(rays_directions, axis=-1, keepdims=True)
    rays_directions = rays_directions / norms

    return rays_sources, rays_directions


def compute_rays_interactions(surfaces: list[SurfaceAbs],
                              rays_sources: np.ndarray,
                              rays_directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the interactions of rays with 3D objects and finds the closest intersection points.


    :param surfaces: List of SurfaceAbs objects representing the 3D surfaces in the scene.
    :param rays_sources: ndarray of shape (N, 3) representing the source coordinates of N rays.
    :param rays_directions: ndarray of shape (N, 3) representing the direction vectors of N rays.

    :return: Tuple containing:
        - ray_interactions: ndarray of shape (N, 3) with the closest intersection points between rays and surfaces.
        - index_list: ndarray of shape (N,) with indices of the surfaces that the rays intersected with.
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


def get_closest_hits(rays_sources: Matrix, rays_directions: Matrix, surfaces: list[SurfaceAbs]) \
        -> tuple[Matrix, Matrix]:
    rays_interactions, index_list = compute_rays_interactions(surfaces, rays_sources, rays_directions)
    return rays_interactions, index_list


def compute_reflection_rays(rays_directions: np.ndarray, surface_normals: np.ndarray) -> np.ndarray:
    """
    Calculate the reflected ray directions for multiple rays given their hit locations and corresponding normals.
    important: reflection_rays is from_shooting_point_to_surfaces iff rays_directions is from_shooting_point_to_surfaces
                i.e. incoming rays -> incoming reflection, outgoing rays -> outgoing reflection

    :param rays_directions: A 2D array of ray directions (shape: [N, 3]).
    :param surface_normals: A 2D array of the surface normals on ray impact point (shape: [N, 3]).
    :return: A 2D array of reflected ray directions (shape: [N, 3]).
    """
    norms = np.linalg.norm(surface_normals, axis=-1, keepdims=True)
    surface_normals = surface_normals / norms

    dot_products = np.sum(rays_directions * surface_normals, axis=-1, keepdims=True)

    reflected_rays = 2 * dot_products * surface_normals - rays_directions

    return reflected_rays
