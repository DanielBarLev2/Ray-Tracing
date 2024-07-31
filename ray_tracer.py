import argparse
import numpy as np
from util import *
from Ray import Ray
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

X_DIRECTION = np.array([1, 0, 0])
Y_DIRECTION = np.array([0, 1, 0])
Z_DIRECTION = np.array([0, 0, 1])


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
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
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

    planes = []
    surfaces = []
    light_sources = []
    for obj in objects:
        if isinstance(obj, InfinitePlane):
            obj.transform_to_camera(view_matrix=view_matrix)
            planes.append(obj)

        elif isinstance(obj, Object3D):
            obj.transform_to_camera(view_matrix=view_matrix)
            surfaces.append(obj)

        elif isinstance(obj, Light):
            obj.transform_to_camera(view_matrix=view_matrix)
            light_sources.append(obj)

    # 6.1.2: Construct a ray from the camera through that pixel
    ray_directions = get_ray_vectors(camera, image_width=args.width, image_height=args.height)

    # 6.2: Check the intersection of the ray with all surfaces in the scene
    ray_sources = np.full_like(ray_directions, camera.position)

    rays_interactions = []

    rays_interactions_planes = []
    for plane in planes:
        plane_intersection = plane.intersect_vectorized(ray_sources=ray_sources, ray_directions=ray_directions)
        rays_interactions_planes.append(plane_intersection)

    rays_interactions.append(rays_interactions_planes)

    rays_interactions_object3d = []
    for obj in objects:
        pass

    # bsp_root = BSPNode.build_bsp_tree(surfaces=surfaces)
    # print(bsp_root)
    # rays_interactions_object3d = traverse(ray_source=ray_sources,
    #                                       ray_directions=ray_directions,
    #                                       bsp_node=bsp_root,
    #                                       rays_interactions=rays_interactions)

    rays_interactions.append(rays_interactions_object3d)

    # 6.3: Find the nearest intersection of the ray. This is the surface that will be seen in the image.
    hit_rays = z_buffer(ray_interactions=rays_interactions)  # ??

    # Dummy result
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


# @todo: test z-buffer
def z_buffer(ray_interactions: list[list[np.ndarray]]) -> np.ndarray:
    """
    Compare this distance with the current value in the z-buffer at the corresponding pixel location.
     If the new distance is smaller (the intersection point is closer to the camera),
      update the z-buffer with this new distance.
    :param ray_interactions: a list of interactions between all rays and every object in the scene.
    :return: the nearest interaction for each ray with any object.
    """
    stacked_arrays = np.stack(*ray_interactions)  # @todo: stack list of lists, not list.

    min_z_indices = np.argmin(stacked_arrays[..., 2], axis=0)

    # Generate grid indices for the last two dimensions
    i, j = np.ogrid[:stacked_arrays.shape[1], :stacked_arrays.shape[2]]

    # Use advanced indexing to select the required entries
    z_buffered = stacked_arrays[min_z_indices, i, j]

    return z_buffered


# def calculate_surface_color(surface: SurfaceAbs, ray: Ray, intersection_point: Vector, light_sources: list[Light]):
#     total_light = calculate_light_on_point(intersection_point, lights=light_sources)
#     ray_tracing(ray)
#
# def ray_tracing(initial_ray: Ray):
#     return None
#
# def calculate_light_on_point(point: Vector, lights: list[Light]):
#     total_light: ColorVector = np.array([0, 0, 0, 0])
#
#     def closest_surface(point: Vector, ray: Ray):
#         return (None, None)
#
#     for light_source in lights:
#         ray: Ray = Ray(ray_source=point, ray_direction=light_source.position - point)
#         (surface, t) = closest_surface(point, ray)
#         if surface is None:
#             total_light += np.append(light_source.color, 1)
#     total_light = (total_light / total_light[3])[:2]
#
#     return total_light


if __name__ == '__main__':
    main()
