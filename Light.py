from util import *
from ray_functions import compute_reflection_rays
from surfaces import SurfaceAbs
from surfaces.Object3D import Object3D
from BSPNode import BSPNode


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
    light_positions_expanded = light_positions[np.newaxis, :, :]
    sources_expanded = sources[:, np.newaxis, :]

    directions = light_positions_expanded - sources_expanded
    lengths = np.linalg.norm(directions, axis=-1, keepdims=True)
    normalized_directions = directions / lengths
    normalized_directions = np.transpose(normalized_directions, (1, 0, 2))

    return normalized_directions


def get_light_base_colors(lights: list[Light], light_directions: np.ndarray, hits: Matrix, shadow_rays_count: int,
                          up_vector: np.ndarray, bsp_tree: BSPNode):
    """
    Calculates the base color and specular intensity contributions of multiple light sources on surfaces.

    :param lights: List of Light objects, each containing properties such as position, color, specular intensity, and shadow intensity.
    :param light_directions: 2D ndarray of shape (num_lights * height * width, 3) representing the direction vectors
                             from each light source to each point on the surfaces.
    :param hits: Matrix representing the hit points on the surfaces.
    :param shadow_rays_count: Number of shadow rays used to compute light intensities and shadows.
    :param up_vector: Vector used for calculating shadow ray deviations.
    :param bsp_tree: BSP tree for efficiently checking intersections and computing shadows.

    :return: Tuple containing two ndarrays:
        - light_color: Array with the same shape as `hits`, representing the color contributions from each light source.
        - light_specular: Array with the same shape as `hits`, representing the specular intensity contributions from each light source.
    """
    lights_diffusive = []
    lights_specular = []
    for i, light in enumerate(lights):
        light_direction = light_directions[i]
        # light intensity
        light_intensity = compute_shadows(light=light, light_direction=-light_direction, hits=hits,
                                          shadow_rays_count=shadow_rays_count, up_vector=up_vector,
                                          bsp_tree=bsp_tree).clip(0, 1)

        # light specular induced color
        specular = light.specular_intensity * light_intensity * light.color
        lights_specular.append(specular)

        # light diffusive induced color
        light_diffusive = light_intensity * light.color
        lights_diffusive.append(light_diffusive)

    lights_diffusive = np.array(lights_diffusive)
    lights_specular = np.array(lights_specular)
    return lights_diffusive, lights_specular


def compute_diffuse_color(obj_diffuse_color: Matrix, light_diffuse_color: Matrix, light_directions: Matrix,
                          surfaces_normals: Matrix):
    """
    Computes the diffuse color contribution of surfaces illuminated by light sources.

    Specular color formula: Sum { Kd * (Lm * V) * Imd } for m in lights
    Kd is diffusive color constant of the object, the ratio of reflection of the specular term of incoming light
    Lm is the direction vector from the point on the surface toward each light source
    N is the normal at this point on the surface
    Ims is the light specular intensity

    :param obj_diffuse_color: Matrix of shape (h * w, 3) representing the diffuse color of the object.
    :param light_diffuse_color: Matrix of shape (N * h * w, 3) representing the diffuse color of each light source.
    :param light_directions: Matrix of shape (N * h * w, 3) representing the direction vectors from the surface to each light source.
    :param surfaces_normals: Matrix of shape (h * w, 3) representing the normal vectors at each point on the surface.

    :return: Matrix of shape (h * w, 3) representing the computed diffuse colors of the surface.
    """
    Kd = obj_diffuse_color
    Lm = light_directions
    N = surfaces_normals
    Imd = light_diffuse_color

    Lm_dot_N = np.sum(Lm * N, axis=-1, keepdims=True)  # Shape: (N, h*w, 1)
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
    Computes the specular color contribution of surfaces illuminated by light sources.

    Specular color formula: Sum { Ks * (Rm * V)^α * Ims } for m in lights
    Ks is specular reflection constant, the ratio of reflection of the specular term of incoming light
    Lm is the direction vector from the point on the surface toward each light source
    Rm is the direction of the reflected ray of light at this point on the surface
    V is the direction pointing towards the viewer (such as a virtual camera).
    α is shininess constant, which is larger for surfaces that are smoother and more mirror-like.
        When this constant is large the specular highlight is small.
    Ims is the light specular intensity

    :param surfaces_specular_color: Matrix of shape (h * w, 3) representing the specular color of the surface.
    :param surfaces_phong_coefficient: Matrix of shape (h * w) representing the Phong shininess coefficient for the surface.
    :param surfaces_to_lights_directions: Matrix of shape (N * h * w, 3) representing the direction vectors from the surface to each light source.
    :param viewer_directions: Matrix of shape (h * w, 3) representing the direction vectors from the surface to the viewer (camera).
    :param surface_normals: Matrix of shape (h * w, 3) representing the normal vectors at each point on the surface.
    :param lights_specular_intensity: Matrix of shape (N * h * w) representing the specular intensity of each light source.

    :return: Matrix of shape (h * w, 3) representing the computed specular colors of the surface.
    """

    Ks = surfaces_specular_color
    Lm = surfaces_to_lights_directions
    Rm = compute_reflection_rays(rays_directions=Lm, surface_normals=surface_normals)

    V = viewer_directions
    alpha = surfaces_phong_coefficient[..., np.newaxis]
    Ims = lights_specular_intensity

    Rm_dot_V = np.sum(Rm * V, axis=-1, keepdims=True)

    specular_colors = np.sum(Ks * (Rm_dot_V ** alpha) * Ims, axis=0)

    specular_colors = np.nan_to_num(specular_colors, nan=0.0)

    return specular_colors


def compute_shadows(light: Light, light_direction: Matrix, hits: Matrix, shadow_rays_count: int, up_vector: np.ndarray,
                    bsp_tree: BSPNode) -> np.ndarray:
    """
    Computes shadow intensity by casting rays to check for surfaces between the light and each pixel.

    This function takes into account the shadow intensity of the light source and the number of shadow rays used.

    :param light: Light object representing the light source, including shadow properties such as shadow intensity.
    :param light_direction: Matrix of shape (N * h * w, 3) representing the direction vectors from the surface to the light source.
    :param hits: Matrix of shape (N * h * w, 3) representing the hit points on the surfaces.
    :param shadow_rays_count: Integer specifying the number of shadow rays to be cast to determine shadow intensity.
    :param bsp_tree: BSPNode object representing the spatial partitioning tree for efficient ray-surface intersection tests.

    :return: ndarray of shape (N * h * w) representing the shadow intensity for each pixel. The values range from
             1.0 (fully lit) to 0.0 (fully shadowed), adjusted based on the light's shadow intensity.
    """
    if light.shadow_intensity == 0:
        return np.ones_like(hits)

    # n - number of pixels, c - color axis(3), s - shadows count
    n, c = hits.shape
    s = shadow_rays_count

    shadow_sources, shadow_rays = get_shadow_rays(light=light, shadows_rays=s,
                                                  light_to_surfaces_directions=light_direction, hits=hits,
                                                  up_vector=up_vector)

    # calculate for each shadow ray the first item it hits
    shadow_sources = shadow_sources.reshape((n * s * s, c))
    shadow_rays = shadow_rays.reshape((n * s * s, c))
    light_hits, l_hits_indices = bsp_tree.intersect_vectorize(ray_sources=shadow_sources, ray_directions=shadow_rays)

    light_hits: Matrix = light_hits.reshape((n, s, s, c))

    # for each pixel calculate percentage of shadow rays that hit it directly
    multi_dim_hits: Matrix = np.tile(hits[:, np.newaxis, np.newaxis, :], (1, s, s, 1))
    dist_from_orig_hits: Matrix = np.linalg.norm((light_hits - multi_dim_hits), axis=-1)
    direct_light_mask: Matrix = dist_from_orig_hits < EPSILON
    direct_light_mask = direct_light_mask.astype(int)[:, :, :, np.newaxis]
    direct_hits_percentage = np.sum(direct_light_mask, axis=(1, 2)) / (s * s)

    # calculate shadow intensity for each pixel
    shadow_intensity = ((1.0 - light.shadow_intensity) + light.shadow_intensity * direct_hits_percentage)
    return shadow_intensity


def get_shadow_rays(light: Light, shadows_rays: int, light_to_surfaces_directions: Matrix, hits: Matrix, up_vector) -> \
        tuple[np.ndarray, np.ndarray]:
    """
    Compute shadow rays originating from the light source.
    Generates shadow rays from the light source for each pixel in the shadow map.
    The rays are adjusted with random deviations to simulate a more realistic shadow.
    
    :param light: Light object representing the light source.
    :param shadows_rays: Number of shadow rays to be cast.
    :param light_to_surfaces_directions: Matrix representing directions from the light to the surface.
    :param hits: Matrix of hit points on the surfaces.
    
    :return: Tuple containing:
        - ray_sources: ndarray of shape (N, shadows_rays, shadows_rays, 3) with the starting points of shadow rays.
        - ray_vectors: ndarray of shape (N, shadows_rays, shadows_rays, 3) with the direction vectors of shadow rays.

    @pre light_to_surfaces_directions are normalized.

    """

    h = w = light.radius
    shadows_rays = n = int(shadows_rays)
    granularity = h / shadows_rays

    light_sources = np.full_like(light_to_surfaces_directions, light.position)
    light_to_surfaces_directions, up, right = diagonalize_vectors(light_to_surfaces_directions, up_vector)
    # Compute pixel centers
    screen_center = light_sources
    pixel_0_0_centers = screen_center + ((h - granularity) / 2 * up) - ((w - granularity) / 2 * right)

    # Generate shadow rays
    i_indices = np.arange(shadows_rays)
    j_indices = np.arange(shadows_rays)
    jj, ii = np.meshgrid(j_indices, i_indices, indexing='ij')

    # Expand dimensions to match the shape of light_sources
    pixel_0_0_centers = np.tile(np.expand_dims(pixel_0_0_centers, axis=(1, 2)), (1, n, n, 1))
    up = np.tile(np.expand_dims(up, axis=(1, 2)), (1, n, n, 1))
    right = np.tile(np.expand_dims(right, axis=(1, 2)), (1, n, n, 1))

    # Calculate ray sources
    ray_sources = pixel_0_0_centers - (ii[..., np.newaxis] * granularity * up) + (
            jj[..., np.newaxis] * granularity * right)

    # Add deviations
    up_deviation_matrix = np.random.uniform(-granularity, granularity, size=ray_sources.shape) * up
    right_deviation_matrix = np.random.uniform(-granularity, granularity, size=ray_sources.shape) * right
    ray_sources += up_deviation_matrix + right_deviation_matrix

    # Calculate ray vectors
    ray_targets = np.tile(np.expand_dims(hits, axis=(1, 2)), (1, n, n, 1))
    ray_vectors = ray_targets - ray_sources

    # Normalize ray vectors
    norms = np.linalg.norm(ray_vectors, axis=-1, keepdims=True)
    ray_vectors = ray_vectors / (norms + EPSILON)

    return ray_sources, ray_vectors
