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


def parse_scene_file(file_path):
    index = 0
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
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
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

    surfaces = []
    light_sources = []
    for obj in objects:
        if isinstance(obj, SurfaceAbs):
            obj.transform_to_camera(view_matrix=view_matrix)
            surfaces.append(obj)

        elif isinstance(obj, Light):
            obj.transform_to_camera(view_matrix=view_matrix)
            light_sources.append(obj)

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

    ray_tracing(rays_sources, rays_directions, surfaces, scene_settings)
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

    return ray_vectors


def compute_rays_interactions(surfaces, rays_sources, rays_directions) -> tuple[list[list], list]:
    rays_interactions = []
    index_list = []

    for surface in surfaces:
        surface_intersection = surface.intersect_vectorized(rays_sources=rays_sources, rays_directions=rays_directions)
        rays_interactions.append(surface_intersection)
        index_list.append(surface.index)

    return rays_interactions, index_list


def calc_ray_hits(ray_interactions: list[list[np.ndarray]], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare this distance with the current value in the z-buffer at the corresponding pixel location.
     If the new distance is smaller (the intersection point is closer to the camera),
      update the z-buffer with this new distance.
    :param ray_interactions: a list of interactions between all rays and every object in the scene.
    :param indices: a list of indices, same size as ray_interactions, for each interaction signify it's object.
    :return: the nearest interaction for each ray with any object.
    """


    "************************************* list is not nested *********************************************"

    flat = [arr for sublist in ray_interactions for arr in sublist]
    stacked_arrays = np.stack(flat)  # @todo: stack list of lists, not list.

    z_values = stacked_arrays[..., 2]

    # Create a mask to identify NaN values
    nan_mask = np.isnan(z_values)

    # Replace NaNs with a large number to effectively ignore them
    # For the purpose of comparison, we use a very large number
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
                scene: SceneSettings):
    """
       Performs ray tracing for a given set of initial rays, calculating interactions with objects,
       and computing both reflected and go-through ray directions.
       This function is designed to handle the interaction of rays with infinite planes, spheres and cubes.

       :param rays_sources: A 3D array of ray sources (shape: [N, N, 3]). Each entry contains the origin of a ray.
       :param rays_directions: A 3D array of ray directions (shape: [N, N, 3]). Each entry contains the direction vector of a ray.
       :param surfaces: A list of 3d objects that might be hit by rays. These objects are used to fetch normals and calculate reflections.
       :param scene: A `SceneSettings` object containing settings for the scene, such as lighting, camera position, etc.

       :return:A 3D array representing the image result
           The function does not return any value. It is intended to perform computations and updates related to ray tracing.
           """

    rays_interactions, interaction_indices = compute_rays_interactions(surfaces=surfaces,
                                                                       rays_sources=rays_sources,
                                                                       rays_directions=rays_directions)

    hits, obj_indices = calc_ray_hits(ray_interactions=rays_interactions, indices=interaction_indices)

    # Calculate reflected, go_through rays - todo: add go_through rays
    reflection_rays_directions = calc_reflection_rays(rays_directions, hits, obj_indices, surfaces)


    # todo continue function
    return


def calc_reflection_rays(inner_rays_directions: np.ndarray, hits: np.ndarray,
                         hits_object_indices: np.ndarray, objects: list) -> np.ndarray:
    """
    Calculate the reflected ray directions for multiple rays given their hit locations and corresponding normals.

    :param inner_rays_directions: A 3D array of ray directions (shape: [N, N, 3]).
    :param hits: A 3D array of hit locations (shape: [N, N, 3]).
    :param hits_object_indices: A 2D array of indices to fetch objects (shape: [N, N]).
    :param objects: A list of objects, each with a method `get_normal_at(hit_location)` to calculate normals.
    :return: A 3D array of reflected ray directions (shape: [N, N, 3]).
    """
    N, M, _ = inner_rays_directions.shape

    # Create an empty array to store normals for each hit location
    normals = np.zeros((N, M, 3))

    # Vectorized fetching of normals
    for idx in np.unique(hits_object_indices):
        obj: SurfaceAbs = objects[idx]
        mask = (hits_object_indices == idx)
        hit_locations = hits[mask]
        normals[mask] = np.array([obj.calculate_normal(loc) for loc in hit_locations])

    # Normalize normals
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / norms

    # Compute dot products
    dot_products = np.sum(inner_rays_directions * normals, axis=-1, keepdims=True)

    # Compute reflected rays
    reflected_rays = inner_rays_directions - 2 * dot_products * normals

    return reflected_rays


if __name__ == '__main__':
    main()
