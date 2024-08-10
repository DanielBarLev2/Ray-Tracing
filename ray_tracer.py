from datetime import datetime

import cv2
import argparse

from BSPNode import BSPNode
from util import *
from PIL import Image
from Camera import Camera
from surfaces.Cube import Cube
from surfaces.Sphere import Sphere
from SceneSettings import SceneSettings
from surfaces.Background import Background
from surfaces.InfinitePlane import InfinitePlane
from Material import Material, get_materials_base_colors
from Light import Light, get_light_base_colors, compute_light_rays, compute_reflection_rays, compute_specular_colors, \
    compute_diffuse_color
from ray_functions import get_initial_rays, get_closest_hits
from surfaces.SurfaceAbs import SurfaceAbs, get_surfaces_normals, get_surfaces_material_indices


def parse_scene_file(file_path):
    index = 1
    mat_index = 1
    objects_3D = []
    surfaces: list[SurfaceAbs] = [Background()]
    materials: list[Material] = []
    lights: list[Light] = []
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
                materials.append(Material(diffuse_color=scene_settings.background_color, specular_color=[0, 0, 0],
                                          reflection_color=[0, 0, 0], shininess=0, transparency=0, mat_index=0))
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10], mat_index)
                mat_index += 1
                materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]), index)
                index += 1
                surfaces.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]), index)
                index += 1
                surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]), index)
                index += 1
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8], index)
                index += 1
                lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
        materials.sort(key=lambda m: m.index)
    return camera, scene_settings, objects_3D, surfaces, materials, lights


def save_image(image_array: np.ndarray, path: str) -> None:
    """

    :param image_array:
    :param path:
    :return:
    """
    now = datetime.now()
    date_str = now.strftime("%m-%d_%H-%M")
    full_path = f"{path}/image_{date_str}.png"

    # Convert the image array to an 8-bit unsigned integer array
    image = Image.fromarray(np.uint8(image_array))

    # Save the image
    image.save(full_path)


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects, surfaces, materials, light_sources = parse_scene_file(args.scene_file)

    bsp_tree = BSPNode.build_bsp_tree(surfaces[4:])
    print(bsp_tree)

    # 6.1.1: Discover the location of the pixel on the camera’s screen
    # 6.1.2: Construct a ray from the camera through that pixel
    rays_sources, rays_directions = get_initial_rays(camera, image_width=args.width, image_height=args.height)

    # 6.2: Check the intersection of the ray with all surfaces in the scene
    image_colors = ray_tracing(rays_sources=rays_sources, rays_directions=rays_directions, surfaces=surfaces,
                               materials=materials, lights=light_sources, scene=scene_settings, camera=camera)

    image_colors = image_colors.reshape((args.height, args.width, 3))
    image_colors = (image_colors * 255).astype(np.uint8)

    print("done")

    bgr_image = cv2.cvtColor(image_colors, cv2.COLOR_RGB2BGR)
    cv2.imshow('RGB Image', bgr_image)
    cv2.waitKey(0)

    # Save the output image
    save_image(image_array=image_colors, path=args.output_image)


def ray_tracing(rays_sources: np.ndarray,
                rays_directions: np.ndarray,
                surfaces: list[SurfaceAbs],
                materials: list[Material],
                lights: list[Light],
                scene: SceneSettings, camera: Camera):
    """
    Performs ray tracing for a given set of initial rays, calculating interactions with objects,
    and computing both reflected and go-through ray directions.
    This function is designed to handle the interaction of rays with infinite planes, spheres and cubes.

    :param camera:
    :param rays_sources: matrix of ray source coordinates.
    :param rays_directions: matrix of ray direction coordinates.
    :param surfaces: a list of 3d objects. used to fetch normals and calculate reflections.
    :param materials: a list of object materials.
    :param lights: a list of light sources in the scene.
    :param scene: a SceneSettings object containing settings for the scene, such as lighting, camera position, etc.
    :return:a 3D array representing the image result
    """
    print(scene.max_recursions)
    if scene.max_recursions <= 0:
        return np.full_like(rays_sources, scene.background_color)

    image_colors = np.full_like(rays_sources, scene.background_color)
    recursion_scene = SceneSettings(scene.background_color, scene.root_number_shadow_rays, scene.max_recursions - 1)

    # 6.2: Check the intersection of the ray with all surfaces in the scene
    # 6.3: Find the nearest intersection of the ray. This is the surface that will be seen in the image
    ray_hits, surfaces_indices = get_closest_hits(rays_sources=rays_sources, rays_directions=rays_directions,
                                                  surfaces=surfaces)

    bg_pixels = (surfaces_indices == 0)
    ray_hits = ray_hits[~bg_pixels]
    rays_directions = rays_directions[~bg_pixels]
    surfaces_indices = surfaces_indices[~bg_pixels]

    surfaces_normals = get_surfaces_normals(surfaces=surfaces, surfaces_indices=surfaces_indices, ray_hits=ray_hits)
    material_indices = get_surfaces_material_indices(surfaces=surfaces, surfaces_indices=surfaces_indices)
    material_colors = get_materials_base_colors(materials=materials, material_indices=material_indices)
    obj_diffusive_colors, obj_specular_colors, phong, reflective_colors, transparency_values = material_colors

    # 6.4.1: Go over each light in the scene.
    surfaces_to_lights_directions = compute_light_rays(ray_hits, lights)
    lights_diffusive, light_specular_intensity = get_light_base_colors(lights=lights,
                                                                       light_directions=surfaces_to_lights_directions,
                                                                       surfaces=surfaces,
                                                                       hits=ray_hits,
                                                                       shadow_rays_count=scene.root_number_shadow_rays)

    # 6.4.2: Add the value it induces on the surface.
    diffusive_colors = compute_diffuse_color(obj_diffuse_color=obj_diffusive_colors,
                                             light_diffuse_color=lights_diffusive,
                                             surfaces_normals=surfaces_normals,
                                             light_directions=surfaces_to_lights_directions)

    specular_colors = compute_specular_colors(surfaces_specular_color=obj_specular_colors,
                                              surfaces_phong_coefficient=phong,
                                              surfaces_to_lights_directions=surfaces_to_lights_directions,
                                              viewer_directions=-rays_directions,
                                              surface_normals=surfaces_normals,
                                              lights_specular_intensity=light_specular_intensity)

    # Additive Colors:
    # output_color = (background_color * transparency) + (diffuse + specular)*(~transparency) + reflection_color
    non_transparency_values = 1.0 - transparency_values
    base_colors = (diffusive_colors + specular_colors) * non_transparency_values

    transparent_mask = (np.any(transparency_values != 0, axis=-1))
    go_through_colors = np.zeros_like(ray_hits)
    if np.any(transparent_mask):
        go_through_rays_directions = rays_directions[transparent_mask]
        go_through_sources = (ray_hits[transparent_mask] + EPSILON * go_through_rays_directions)
        go_through_colors[transparent_mask] = ray_tracing(rays_sources=go_through_sources,
                                                          rays_directions=go_through_rays_directions, surfaces=surfaces,
                                                          materials=materials, lights=lights, scene=recursion_scene,
                                                          camera=camera)
    back_colors = go_through_colors * transparency_values

    reflective_mask = (np.linalg.norm(reflective_colors, axis=-1) != 0)
    reflection = np.zeros_like(ray_hits)
    if np.any(reflective_mask):
        reflection_rays_directions = compute_reflection_rays(-rays_directions[reflective_mask],
                                                             surfaces_normals[reflective_mask])
        reflection_rays_sources = (ray_hits[reflective_mask] + EPSILON * reflection_rays_directions)
        reflection[reflective_mask] = ray_tracing(rays_sources=reflection_rays_sources,
                                                  rays_directions=reflection_rays_directions, surfaces=surfaces,
                                                  materials=materials,
                                                  lights=lights, scene=recursion_scene, camera=camera)
    reflection *= reflective_colors

    image_colors[~bg_pixels] = (back_colors + base_colors + reflection)
    image_colors[image_colors > 1] = 1
    return image_colors


if __name__ == '__main__':
    main()
