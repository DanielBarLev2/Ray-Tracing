from util import *
from surfaces import SurfaceAbs
from SceneSettings import SceneSettings
from surfaces.Object3D import Object3D
from ray_functions import compute_rays_interactions, compute_rays_hits, get_closest_hits


class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius, index: int):
        self.position = np.array(position)
        self.color = np.array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius
        self.index = index

    def __repr__(self):
        return f"Light:\n" \
               f"pos: {self.position}\n" \
               f"color: {self.color}\n" \
               f"spec: {self.specular_intensity}\n" \
               f"shadow: {self.shadow_intensity}\n" \
               f"radius: {self.radius}\n" \
               f"idx: {self.index}\n"

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
    :param scene: containing the number of shadow root rays.
    :return: A tuple containing two numpy arrays:
             - light_color: An array with the same shape as hits, representing the color contributions from the lights.
             - light_specular: An array with the same shape as hits, representing the specular lights.
    """
    lights_diffusive = []
    lights_specular = []
    direct_lights_masks = []
    for i, light in enumerate(lights):
        light_direction = light_directions[i]
        light_sources = np.full_like(light_direction, light.position)
        l_hits, _ = get_closest_hits(rays_sources=light_sources, rays_directions=-light_direction, surfaces=surfaces)

        # light intensity
        light_intensity = compute_shadows(light=light, surfaces=surfaces, light_sources=light_sources,
                                          light_direction=-light_direction,
                                          hits=hits, scene=scene).clip(0, 1)

        # light specular induced color
        specular = (light.specular_intensity * light.color) * light_intensity
        lights_specular.append(specular)

        # light diffusive induced color
        light_diffusive = (light_intensity * light.color)
        lights_diffusive.append(light_diffusive)

    lights_diffusive = np.array(lights_diffusive)
    lights_specular = np.array(lights_specular)
    return lights_diffusive, lights_specular


def compute_diffuse_color(obj_diffuse_color: Matrix, light_diffuse_color: Matrix, light_directions: Matrix,
                          surfaces_normals:Matrix):
    """
        Specular color formula: Sum { Kd * (Lm * V) * Imd } for m in lights
        Kd is diffusive color constant of the object, the ratio of reflection of the specular term of incoming light
        Lm is the direction vector from the point on the surface toward each light source
        N is the normal at this point on the surface
        Ims is the light specular intensity"""
    Kd = obj_diffuse_color
    Lm = light_directions
    N = surfaces_normals
    Imd = light_diffuse_color

    Lm_dot_N = np.sum(Lm * N, axis=-1, keepdims=True)  # Shape: (N, h, w)
    Lm_dot_N = np.maximum(Lm_dot_N, 0)  # to remove negative values
    diffuse_colors = np.sum(Kd[np.newaxis, ...] * Lm_dot_N * Imd, axis=0)
    return diffuse_colors


def compute_specular_colors(surfaces_specular_color: Matrix,
                            surfaces_phong_coefficient: Matrix,
                            surfaces_to_lights_directions: np.ndarray,
                            viewer_directions: Matrix,
                            surface_normals: Matrix,
                            lights_specular_intensity: np.ndarray):
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
    Ims = lights_specular_intensity

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
                    light_sources: Matrix,
                    light_direction: Matrix,
                    hits: Matrix,
                    scene: SceneSettings):
    shadow_sources, shadow_rays = get_shadow_rays(light=light,
                                                  shadows_rays=scene.root_number_shadow_rays,
                                                  normals_rays=light_direction, hits=hits)
    h, w, d = hits.shape
    n = int(scene.root_number_shadow_rays)

    shadow_sources = shadow_sources.reshape((h * n, w * n, d))
    shadow_rays = shadow_rays.reshape((h * n, w * n, d))
    light_hits, l_hits_object_indices = get_closest_hits(surfaces=surfaces, rays_sources=shadow_sources, rays_directions=shadow_rays)
    light_hits: Matrix = light_hits.reshape((h, w, n, n, d))
    l_hits_object_indices = l_hits_object_indices.reshape((h, w, n, n))
    background_hits = l_hits_object_indices == 0

    light_hits[light_hits == np.inf] = -1
    hits[hits == np.inf] = -1

    multi_dim_hits: Matrix = np.tile(hits[:, :, np.newaxis, np.newaxis, :], (1, 1, n, n, 1))
    dist_from_orig_hits: Matrix = np.linalg.norm((light_hits - multi_dim_hits), axis=-1)
    direct_light_mask: Matrix = dist_from_orig_hits < EPSILON
    direct_light_mask[background_hits] = 1

    direct_light_mask = direct_light_mask.astype(int)[:, :, :, :, np.newaxis]
    direct_hits_percentage = np.sum(direct_light_mask, axis=(2, 3)) / (n * n)
    shadow_intensity = ((1.0 - light.shadow_intensity) + light.shadow_intensity * direct_hits_percentage)
    return shadow_intensity


def get_shadow_rays(light: Light, shadows_rays: int, normals_rays: Matrix, hits: Matrix) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Compute shadow rays originating from the light source.
    :param light: The light source.
    :param shadows_rays: Number of shadow rays.
    :param normals_rays: Normal vectors at the hit points.
    :param hits: Hit points on the surfaces.
    :return: Tuple of ray sources and ray vectors.
    """

    h = w = light.radius
    shadows_rays = n = int(shadows_rays)
    granularity = h / shadows_rays

    light_sources = np.full_like(normals_rays, light.position)
    normals_rays = normals_rays / np.linalg.norm(normals_rays, axis=2, keepdims=True)

    up = Y_DIRECTION
    if np.any(np.all(np.cross(normals_rays, up) < EPSILON, axis=-1)):
        up = Z_DIRECTION

    # Compute up and right vectors
    up_projection = np.sum(up * normals_rays, axis=-1, keepdims=True)
    up = up - up_projection * normals_rays
    up = up / (np.linalg.norm(up, axis=-1, keepdims=True) + EPSILON)

    right = np.cross(up, normals_rays)
    right = right / (np.linalg.norm(right, axis=-1, keepdims=True) + EPSILON)

    # Compute pixel centers
    screen_center = light_sources
    pixel_0_0_centers = screen_center + ((h - granularity) / 2 * up) - ((w - granularity) / 2 * right)

    # Generate shadow rays
    i_indices = np.arange(shadows_rays)
    j_indices = np.arange(shadows_rays)
    jj, ii = np.meshgrid(j_indices, i_indices, indexing='ij')

    # Expand dimensions to match the shape of light_sources
    pixel_0_0_centers = np.tile(np.expand_dims(pixel_0_0_centers, axis=(2, 3)), (1, 1, n, n, 1))
    up = np.tile(np.expand_dims(up, axis=(2, 3)), (1, 1, n, n, 1))
    right = np.tile(np.expand_dims(right, axis=(2, 3)), (1, 1, n, n, 1))

    # Calculate ray sources
    ray_sources = pixel_0_0_centers - (ii[..., np.newaxis] * granularity * up) + (
            jj[..., np.newaxis] * granularity * right)

    # Add deviations
    up_deviation_matrix = np.random.uniform(-granularity, granularity, size=ray_sources.shape) * up
    right_deviation_matrix = np.random.uniform(-granularity, granularity, size=ray_sources.shape) * right
    ray_sources += up_deviation_matrix + right_deviation_matrix

    # Calculate ray vectors
    ray_targets = np.tile(np.expand_dims(hits, axis=(2, 3)), (1, 1, n, n, 1))
    ray_vectors = ray_targets - ray_sources

    # Normalize ray vectors
    norms = np.linalg.norm(ray_vectors, axis=-1, keepdims=True)
    ray_vectors = ray_vectors / (norms + EPSILON)

    return ray_sources, ray_vectors
