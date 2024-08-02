from util import *
from surfaces import SurfaceAbs
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
                          hits: Matrix):
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
        light_sources = np.ones_like(light_directions[i]) * light.position

        light_rays_interactions, light_index_list = compute_rays_interactions(surfaces=surfaces,
                                                                              rays_sources=light_sources,
                                                                              rays_directions=-light_directions[i])

        light_hits, obj_indices = compute_rays_hits(ray_interactions=light_rays_interactions,
                                                    index_list=light_index_list)

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
