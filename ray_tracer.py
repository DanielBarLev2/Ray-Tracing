import argparse
import numpy as np
from util import *
from PIL import Image
from Light import Light
from Camera import Camera
from BSPNode import BSPNode, traverse
from Material import Material
from surfaces.Cube import Cube
from surfaces.Sphere import Sphere
from surfaces.Object3D import Object3D
from SceneSettings import SceneSettings
from surfaces.SurfaceAbs import SurfaceAbs
from surfaces.InfinitePlane import InfinitePlane
import cv2


def parse_scene_file(file_path):
    index = 0
    mat_index = 0
    objects_3D = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10], mat_index)
                mat_index += 1
                objects_3D.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]), index)
                index += 1
                objects_3D.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]), index)
                index += 1
                objects_3D.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]), index)
                index += 1
                objects_3D.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8], index)
                index += 1
                objects_3D.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects_3D


def save_image(image_array: np.ndarray, path: str) -> None:
    """

    :param image_array:
    :param path:
    :return:
    """
    image = Image.fromarray(np.uint8(image_array))
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # 6.1.1: Discover the location of the pixel on the camera’s screen
    view_matrix = camera.create_view_matrix()
    camera.transform_to_camera(view_matrix=view_matrix)

    surfaces: list[SurfaceAbs] = []
    materials: list[Material] = []
    light_sources = []
    for obj in objects:
        if isinstance(obj, SurfaceAbs):
            obj.transform_to_camera(view_matrix=view_matrix)
            surfaces.append(obj)

        elif isinstance(obj, Light):
            obj.transform_to_camera(view_matrix=view_matrix)
            light_sources.append(obj)

        elif isinstance(obj, Material):
            materials.append(obj)

    materials.sort(key=lambda x: x.index)
    # 6.1.2: Construct a ray from the camera through that pixel
    rays_directions = get_ray_vectors(camera, image_width=args.width, image_height=args.height)

    # 6.2: Check the intersection of the ray with all surfaces in the scene
    rays_sources = np.full_like(rays_directions, camera.position)

    rays_interactions, index_list = compute_rays_interactions(surfaces, rays_sources, rays_directions)

    # bsp_root = BSPNode.build_bsp_tree(surfaces=surfaces)
    # print(bsp_root)
    # rays_interactions_object3d = traverse(ray_source=ray_sources,
    #                                       ray_directions=ray_directions,
    #                                       bsp_node=bsp_root,
    #                                       rays_interactions=rays_interactions)

    # 6.3: Find the nearest intersection of the ray. This is the surface that will be seen in the image.
    surfaces = [o for o in objects if isinstance(o, SurfaceAbs)]

    image_colors = (ray_tracing(rays_sources, rays_directions, surfaces, materials, light_sources,
                                scene_settings) * 255).astype(np.uint8)
    bgr_image = cv2.cvtColor(image_colors, cv2.COLOR_RGB2BGR)
    cv2.imshow('RGB Image', bgr_image)
    cv2.waitKey(0)
    image_array = np.zeros((args.width, args.height, 3))

    # Save the output image
    save_image(image_array=image_array, path=args.output_image)


def get_ray_vectors(camera: Camera, image_width: int, image_height: int) -> np.ndarray:
    """
    Generates a 3D array of ray vectors for each pixel in the specified image dimensions based on the camera settings.
    :param camera: the camera object with properties defining its screen size and distance.
    :param image_width: the number of horizontal pixels in the image.
    :param image_height: the number of vertical pixels in the image.
    :return:  A 3D array (image_height, image_width, 3) where each element is a vector [x, y, z] representing
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


def compute_rays_interactions(surfaces, rays_sources, rays_directions) -> tuple[list, list]:
    rays_interactions = []
    index_list = []

    for surface in surfaces:
        surface_intersection = surface.intersect_vectorized(rays_sources=rays_sources, rays_directions=rays_directions)
        rays_interactions.append(surface_intersection)
        index_list.append(surface.index)

    # background index and virtual intersections. todo: Hi Daniel :), decide how to handle this, currently not working
    max_v = np.max(np.nan_to_num(rays_interactions, nan=-np.inf))
    rays_interactions.append(np.full_like(rays_sources, max_v + 1))
    index_list.append(-1)

    return rays_interactions, index_list


def calc_ray_hits(ray_interactions: list[np.ndarray], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare this distance with the current value in the z-buffer at the corresponding pixel location.
     If the new distance is smaller (the intersection point is closer to the camera),
      update the z-buffer with this new distance.
    :param ray_interactions: a list of interactions between all rays and every object in the scene.
    :param indices: a list of indices, same size as ray_interactions, for each interaction signify it's object.
    :return: the nearest interaction for each ray with any object.
    """

    stacked_arrays = np.stack(ray_interactions)

    z_values = stacked_arrays[..., 2]

    # Create a mask to identify NaN values
    nan_mask = np.isnan(z_values)

    # Replace NaNs with a large number to effectively ignore them
    z_values_with_large_number = np.where(nan_mask, np.inf, z_values)

    # Compute the indices of the minimum values, ignoring NaNs
    min_z_indices = np.argmin(z_values_with_large_number, axis=0)
    # Generate grid indices for the last two dimensions
    i, j = np.ogrid[:stacked_arrays.shape[1], :stacked_arrays.shape[2]]

    # Use advanced indexing to select the required entries
    z_buffered = stacked_arrays[min_z_indices, i, j]

    # Create a numpy array from indices list
    indices_array = np.array(indices)

    # Use min_z_indices to get the corresponding object IDs
    object_ids = indices_array[min_z_indices]

    return z_buffered, object_ids


def ray_tracing(rays_sources: Matrix[Vector], rays_directions: Matrix[Vector], surfaces: list[SurfaceAbs],
                materials: list[Material], lights: list[Light], scene: SceneSettings):
    """
       Performs ray tracing for a given set of initial rays, calculating interactions with objects,
       and computing both reflected and go-through ray directions.
       This function is designed to handle the interaction of rays with infinite planes, spheres and cubes.

       :param rays_sources: A 3D array of ray sources (shape: [N, N, 3]). Each entry contains the origin of a ray.
       :param rays_directions: A 3D array of ray directions (shape: [N, N, 3]). Each entry contains the direction vector of a ray.
       :param surfaces: A list of 3d objects that might be hit by rays. These objects are used to fetch normals and calculate reflections.
       :param materials: A list of object materials
       :param lights: A list of light sources in the scene
       :param scene: A `SceneSettings` object containing settings for the scene, such as lighting, camera position, etc.
       :return:A 3D array representing the image result
           The function does not return any value. It is intended to perform computations and updates related to ray tracing.
           """
    recursion_scene = SceneSettings(scene.background_color, scene.root_number_shadow_rays, scene.max_recursions - 1)

    rays_interactions, interaction_indices = compute_rays_interactions(surfaces=surfaces,
                                                                       rays_sources=rays_sources,
                                                                       rays_directions=rays_directions)

    hits, obj_indices = calc_ray_hits(ray_interactions=rays_interactions, indices=interaction_indices)

    surfaces_normals = apply_object_method(surfaces, obj_indices, "calculate_normal", hits.shape, hits)

    material_indices = apply_object_method(objects=surfaces, indices_matrix=obj_indices,
                                           method_name="get_material_index",
                                           res_shape=obj_indices.shape).astype(dtype=int)

    material_colors = get_materials_base_colors(materials, material_indices)
    diffusive_colors, base_specular_colors, phong, reflective_colors, transparency_values = material_colors

    surfaces_to_lights_directions = calc_light_rays(hits, lights)
    light_color, light_specular_intensity = get_light_base_colors(lights, surfaces_to_lights_directions, surfaces, hits)

    specular_colors = calculate_specular_colors(surfaces_specular_color=base_specular_colors,
                                                surfaces_phong_coefficient=phong,
                                                surfaces_to_lights_directions=surfaces_to_lights_directions,
                                                viewer_directions=-rays_directions,
                                                surface_normals=surfaces_normals,
                                                lights=lights, light_specular_intensity=light_specular_intensity)

    # transparency_values = np.zeros_like(hits)
    non_transparency_values = np.ones_like(hits) - transparency_values

    base_colors = (diffusive_colors + specular_colors) * light_color * non_transparency_values

    go_through_rays_directions = rays_directions
    if scene.max_recursions > 0:
        go_through_colors = ray_tracing(hits, go_through_rays_directions, surfaces, materials, lights, recursion_scene)
    else:
        go_through_colors = np.zeros_like(diffusive_colors)
    back_colors = go_through_colors * transparency_values

    reflection_rays_directions = calc_reflection_rays(-rays_directions, surfaces_normals)
    if scene.max_recursions > 0:
        reflection = ray_tracing(hits, reflection_rays_directions, surfaces, materials, lights, recursion_scene)
    else:
        reflection = np.zeros_like(diffusive_colors)
    reflection *= reflective_colors

    image_colors = (back_colors + base_colors + reflection)
    image_colors[image_colors > 1] = 1
    # image_colors = image_colors/np.max(image_colors)
    return image_colors


def get_materials_base_colors(materials: list[Material], material_indices: Matrix[int]):
    res_shape = (*material_indices.shape, 3)
    diffusive_colors = apply_object_method(objects=materials, indices_matrix=material_indices,
                                           method_name="get_diffusive", res_shape=res_shape)

    surfaces_specular_colors = apply_object_method(objects=materials, indices_matrix=material_indices,
                                                   method_name="get_specular", res_shape=res_shape)
    phong = apply_object_method(objects=materials, indices_matrix=material_indices,
                                method_name="get_shininess", res_shape=material_indices.shape)

    reflective_colors = apply_object_method(objects=materials, indices_matrix=material_indices,
                                            method_name="get_reflective", res_shape=res_shape)

    transparency_values = apply_object_method(objects=materials, indices_matrix=material_indices,
                                              method_name="get_transparency", res_shape=res_shape)

    return diffusive_colors, surfaces_specular_colors, phong, reflective_colors, transparency_values


def get_light_base_colors(lights: list[Light], light_directions: list[Matrix[Vector]], surfaces: list[SurfaceAbs],
                          hits: Matrix[Vector]):
    light_color = np.zeros_like(hits)
    light_specular = np.zeros_like(hits)
    for i, light in enumerate(lights):
        light_sources = np.ones_like(light_directions[i]) * light.position

        lr_interactions, l_interaction_indices = compute_rays_interactions(surfaces=surfaces,
                                                                           rays_sources=light_sources,
                                                                           rays_directions=-light_directions[i])
        l_hits, obj_indices = calc_ray_hits(ray_interactions=lr_interactions, indices=l_interaction_indices)

        direct_light_mask = (l_hits - hits) < EPSILON
        obscured_light_mask = 1.0 - direct_light_mask

        direct_light_intensity = direct_light_mask * 1.0
        obscured_light_intensity = obscured_light_mask * (1.0 - light.shadow_intensity)
        light_intensity = np.maximum(direct_light_intensity, obscured_light_intensity)

        light_color += (light.color * light_intensity)
        light_specular += (light.specular_intensity * direct_light_mask)

    light_color = np.clip(light_color, None, 1)
    light_specular = np.clip(light_specular, None, 1)
    return light_color, light_specular


def calc_light_rays(sources: Matrix[Vector], lights: list[Light]) -> list[Matrix[Vector]]:
    """
    calculate rays directions from all sources to each light source.

    :param sources: A 2D matrix of vectors, each vector represents a 3D point in space.
    :param lights: A list of light sources in the scene.
    :return: list of len(lights) matrices. each cell in each matrix is a Vector representing a ray from source to light
    """
    ray_directions = []
    for light in lights:
        direction = (light.position - sources)
        length = np.linalg.norm(direction, axis=2)
        ray_directions.append(direction / length[:, :, np.newaxis])

    return ray_directions


def calculate_specular_colors(surfaces_specular_color: Matrix[ColorVector], surfaces_phong_coefficient: Matrix[float],
                              surfaces_to_lights_directions: list[Matrix[Vector]], viewer_directions: Matrix[Vector],
                              surface_normals: Matrix[Vector], lights: [list[Light]],
                              light_specular_intensity: np.ndarray[Matrix[float]]):
    """
    Specular color formula: Sum { Ks * (Rm * V)^α * Ims } for m in lights
    Ks is specular reflection constant, the ratio of reflection of the specular term of incoming light
    Lm is the direction vector from the point on the surface toward each light source
    Rm is the direction of the reflected ray of light at this point on the surface
    V is the direction pointing towards the viewer (such as a virtual camera).
    α is shininess constant, which is larger for surfaces that are smoother and more mirror-like.
       When this constant is large the specular highlight is small.
    Ims is the light specular intensity"""

    n = len(lights)
    Ks = surfaces_specular_color
    Lm = surfaces_to_lights_directions
    Rm = np.array([calc_reflection_rays(rays_directions=Lm[i], surface_normals=surface_normals) for i in range(n)])
    V = viewer_directions
    alpha = surfaces_phong_coefficient[..., np.newaxis]  # todo: check about coefficient
    Ims = light_specular_intensity
    Rm_dot_V = np.array([np.sum(Rm[i] * V, axis=-1, keepdims=True) for i in range(n)])

    specular_colors = sum([Ks * (Rm_dot_V[i] ** alpha) * Ims for i in range(n)])
    specular_colors = np.nan_to_num(specular_colors, nan=0.0)
    return specular_colors


def apply_object_method(objects: list, indices_matrix: Matrix, method_name: str, res_shape: np.shape,
                        data_matrix: Matrix | None = None):
    res: np.ndarray = np.zeros(res_shape)
    for idx in np.unique(indices_matrix):
        if idx == -1:  # todo: bg - handle this case.
            continue
        obj = objects[idx]
        mask = (indices_matrix == idx)
        if data_matrix is None:
            res[mask] = getattr(obj, method_name)()
        else:
            relevant_data = data_matrix[mask]
            calculation = np.array([getattr(obj, method_name)(data) for data in relevant_data])
            res[mask] = calculation
    return res


def calc_reflection_rays(rays_directions: Matrix[Vector], surface_normals: Matrix[Vector]) -> np.ndarray:
    """
    Calculate the reflected ray directions for multiple rays given their hit locations and corresponding normals.
    important: reflection_rays is from_shooting_point_to_surfaces iff rays_directions is from_shooting_point_to_surfaces
                i.e. incoming rays -> incoming reflection, outgoing rays -> outgoing reflection

    :param rays_directions: A 3D array of ray directions (shape: [N, N, 3]).
    :param surface_normals: A 3D array of the surface normals on ray impact point (shape: [N, N, 3]).
    :return: A 3D array of reflected ray directions (shape: [N, N, 3]).
    """
    N, M, _ = rays_directions.shape

    # Normalize normals
    norms = np.linalg.norm(surface_normals, axis=-1, keepdims=True)
    surface_normals = surface_normals / norms

    # Compute dot products
    dot_products = np.sum(rays_directions * surface_normals, axis=-1, keepdims=True)

    # Compute reflected rays
    reflected_rays = 2 * dot_products * surface_normals - rays_directions

    return reflected_rays


if __name__ == '__main__':
    main()
