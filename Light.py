import numpy as np

from util import *
from surfaces import SurfaceAbs
from Camera import Camera
from SceneSettings import SceneSettings
from surfaces.Object3D import Object3D
from ray_functions import compute_rays_interactions, compute_rays_hits


class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius, index: int):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
        self.index = index

    def transform_to_camera(self, view_matrix) -> None:
        """
        Transform the position of the light using the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return:
        """
        self.position = Object3D.transform_to_camera_coordinates(self.position, view_matrix)


def compute_light_rays(sources: np.ndarray, lights: list[Light]) -> np.ndarray:
    """
    Calculate rays directions from all sources to each light source.

    :param sources: A 2D matrix of vectors, each vector represents a 3D point in space.
    :param lights: A list of light sources in the scene.
    :return: A 3D numpy array where each slice along the third axis is a matrix of vectors representing
             rays from sources to a specific light.
    """
    light_positions = np.array([light.position for light in lights])
    light_positions_expanded = light_positions[np.newaxis, np.newaxis, :, :]
    sources_expanded = sources[:, :, np.newaxis, :]

    directions = light_positions_expanded - sources_expanded
    lengths = np.linalg.norm(directions, axis=3, keepdims=True)
    normalized_directions = directions / lengths
    normalized_directions = np.transpose(normalized_directions, (2, 0, 1, 3))

    return normalized_directions


def get_light_base_colors(lights: list[Light],
                          light_directions: np.ndarray,
                          surfaces: list[SurfaceAbs],
                          hits: Matrix,
                          camera: Camera,
                          scene: SceneSettings):
    """
    Calculate the base colors and specular values of light sources on surfaces.

    This function computes the color and specular intensity contributions of multiple light sources
    to a surface based on their interactions with the surfaces and the hits detected.

    :param lights: A list of Light objects, each containing properties such as position, color,
                   specular intensity, and shadow intensity.
    :param light_directions: A 4D numpy array with shape (num_lights, height, width, 3) representing
                             the direction vectors from each source to each light.
    :param surfaces: A list of SurfaceAbs objects representing the surfaces in the scene.
    :param hits: A Matrix representing the hit points on the surfaces.
    :return: A tuple containing two numpy arrays:
             - light_color: An array with the same shape as hits, representing the color contributions from the lights.
             - light_specular: An array with the same shape as hits, representing the specular lights.
    """
    light_color = np.zeros_like(hits)
    light_specular = np.zeros_like(hits)

    for i, light in enumerate(lights):
        light_direction = light_directions[i]
        light_sources = np.ones_like(light_direction) * light.position

        compute_shadows(light=light, surfaces=surfaces, rays_sources=light_sources, rays_directions=-light_direction
                        , hits=hits, camera=camera, scene=scene)

        light_rays_interactions, light_index_list = compute_rays_interactions(surfaces=surfaces,
                                                                              rays_sources=light_sources,
                                                                              rays_directions=-light_direction)

        light_hits, obj_indices = compute_rays_hits(ray_interactions=light_rays_interactions,
                                                    index_list=light_index_list)

        light_hits[light_hits == np.inf] = 0
        hits[hits == np.inf] = 0

        direct_light_mask = (light_hits - hits) < EPSILON
        obscured_light_mask = 1.0 - direct_light_mask

        direct_light_intensity = direct_light_mask * 1.0
        obscured_light_intensity = obscured_light_mask * (1.0 - light.shadow_intensity)
        light_intensity = np.maximum(direct_light_intensity, obscured_light_intensity)

        light_color += (light.color * light_intensity)
        light_specular += (light.specular_intensity * direct_light_mask)

    light_color = np.clip(light_color, a_min=None, a_max=1)
    light_specular = np.clip(light_specular, a_min=None, a_max=1)

    return light_color, light_specular


def compute_specular_colors(surfaces_specular_color: Matrix,
                            surfaces_phong_coefficient: Matrix,
                            surfaces_to_lights_directions: np.ndarray,
                            viewer_directions: Matrix,
                            surface_normals: Matrix,
                            light_specular_intensity: np.ndarray):
    """
    Specular color formula: Sum { Ks * (Rm * V)^α * Ims } for m in lights
    Ks is specular reflection constant, the ratio of reflection of the specular term of incoming light
    Lm is the direction vector from the point on the surface toward each light source
    Rm is the direction of the reflected ray of light at this point on the surface
    V is the direction pointing towards the viewer (such as a virtual camera).
    α is shininess constant, which is larger for surfaces that are smoother and more mirror-like.
       When this constant is large the specular highlight is small.
    Ims is the light specular intensity"""

    Ks = surfaces_specular_color
    Lm = surfaces_to_lights_directions
    Rm = compute_reflection_rays(lights_rays_directions=Lm, surface_normals=surface_normals)

    V = viewer_directions
    alpha = surfaces_phong_coefficient[..., np.newaxis]
    Ims = light_specular_intensity

    Rm_dot_V = np.sum(Rm * V, axis=-1, keepdims=True)

    specular_colors = np.sum(Ks * (Rm_dot_V ** alpha) * Ims, axis=0)

    specular_colors = np.nan_to_num(specular_colors, nan=0.0)

    return specular_colors


def compute_reflection_rays(lights_rays_directions: np.ndarray, surface_normals: np.ndarray) -> np.ndarray:
    """
    Calculate the reflected ray directions for multiple rays given their hit locations and corresponding normals.
    important: reflection_rays is from_shooting_point_to_surfaces iff rays_directions is from_shooting_point_to_surfaces
                i.e. incoming rays -> incoming reflection, outgoing rays -> outgoing reflection

    :param lights_rays_directions: A 4D array of ray directions (shape: [L, N, N, 3]).
    :param surface_normals: A 3D array of the surface normals on ray impact point (shape: [N, N, 3]).
    :return: A 3D array of reflected ray directions (shape: [N, N, 3]).
    """
    norms = np.linalg.norm(surface_normals, axis=-1, keepdims=True)
    surface_normals = surface_normals / (norms + EPSILON)

    dot_products = np.sum(lights_rays_directions * surface_normals, axis=-1, keepdims=True)

    reflected_rays = 2 * dot_products * surface_normals - lights_rays_directions

    return reflected_rays


def compute_shadows(light: Light,
                    surfaces: list[SurfaceAbs],
                    rays_sources: Matrix,
                    rays_directions: Matrix,
                    hits: Matrix,
                    camera: Camera,
                    scene: SceneSettings):
    shadow_rays = get_shadow_rays(light=light,
                                  camera=camera,
                                  shadows_rays=scene.root_number_shadow_rays,
                                  normals_rays=rays_directions, hits=hits)


def get_shadow_rays(light: Light, camera: Camera, shadows_rays: int, normals_rays: Matrix, hits: Matrix) -> np.ndarray:
    """

    :param light:
    :param camera:
    :param shadows_rays:
    :param normals_rays:
    :param hits:
    :return:
    """

    h = w = light.radius
    shadows_rays = int(shadows_rays)
    granularity = h / shadows_rays

    light_sources = np.full_like(normals_rays, light.position)
    normals_rays = normals_rays / np.linalg.norm(normals_rays, axis=2, keepdims=True)

    up = camera.up_vector
    if np.any(np.all(np.cross(normals_rays, up) == 0, axis=2)):
        up = camera.look_direction

    right = np.cross(up, normals_rays)
    right = right / np.linalg.norm(right, axis=2, keepdims=True)

    up = np.cross(right, normals_rays)
    up = up / np.linalg.norm(up, axis=2, keepdims=True)

    screen_center = light_sources

    pixel_0_0_centers = screen_center + ((h - granularity) / 2 * up) - ((w - granularity) / 2 * right)

    i_indices = np.arange(shadows_rays)
    j_indices = np.arange(shadows_rays)
    jj, ii = np.meshgrid(j_indices, i_indices)

    # broadcast
    up = np.tile(up[:, :, np.newaxis, np.newaxis, :],
                 (1, 1, shadows_rays, shadows_rays, 1))
    right = np.tile(right[:, :, np.newaxis, np.newaxis, :],
                    (1, 1, shadows_rays, shadows_rays, 1))
    pixel_0_0_centers = np.tile(pixel_0_0_centers[:, :, np.newaxis, np.newaxis, :],
                                (1, 1, shadows_rays, shadows_rays, 1))

    ray_sources = (
            pixel_0_0_centers
            - (ii[:, :, np.newaxis] * granularity * up)
            + (jj[:, :, np.newaxis] * granularity * right))

    up_deviation_matrix = np.random.uniform(-granularity,
                                            granularity,
                                            size=(normals_rays.shape[0],
                                                  normals_rays.shape[1],
                                                  shadows_rays, shadows_rays, 3)) * up

    right_deviation_matrix = np.random.uniform(-granularity,
                                               granularity,
                                               size=(normals_rays.shape[0],
                                                     normals_rays.shape[1],
                                                     shadows_rays,
                                                     shadows_rays, 3)) * right

    ray_sources += up_deviation_matrix + right_deviation_matrix

    ray_vectors = np.tile(hits[:, :, np.newaxis, np.newaxis, :],
                          (1, 1, shadows_rays, shadows_rays, 1)) - ray_sources

    norms = np.linalg.norm(ray_vectors, axis=2, keepdims=True)
    ray_vectors = ray_vectors / norms

    return ray_vectors
