import cv2
import argparse

import numpy as np

from surfaces.Background import Background
from surfaces.Object3D import Object3D
from util import *
from PIL import Image
from Camera import Camera
from surfaces.Cube import Cube
from surfaces.Sphere import Sphere
from SceneSettings import SceneSettings
from surfaces.InfinitePlane import InfinitePlane
from Material import Material, get_materials_base_colors
from Light import Light, get_light_base_colors, compute_light_rays, compute_reflection_rays, compute_specular_colors
from ray_functions import get_ray_vectors, compute_rays_interactions, compute_rays_hits
from surfaces.SurfaceAbs import SurfaceAbs, get_surfaces_normals, get_surfaces_material_indies


def parse_scene_file(file_path):
    index = 1
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
    light_sources: list[Light] = []

    surfaces.append(Background())
    materials.append(Material(diffuse_color=scene_settings.background_color,
                              specular_color=[0, 0, 0],
                              reflection_color=[0, 0, 0],
                              shininess=0,
                              transparency=0,
                              mat_index=0))

    for obj in objects:
        if isinstance(obj, SurfaceAbs):
            obj.transform_to_camera(view_matrix=view_matrix)
            surfaces.append(obj)

        elif isinstance(obj, Light):
            obj.transform_to_camera(view_matrix=view_matrix)
            light_sources.append(obj)

        elif isinstance(obj, Material):
            materials.append(obj)

    # 6.1.2: Construct a ray from the camera through that pixel
    rays_directions = get_ray_vectors(camera, image_width=args.width, image_height=args.height)

    # 6.2: Check the intersection of the ray with all surfaces in the scene
    rays_sources = np.full_like(rays_directions, camera.position)

    materials.sort(key=lambda x: x.index)

    image_colors = ray_tracing(rays_sources=rays_sources,
                               rays_directions=rays_directions,
                               surfaces=surfaces,
                               materials=materials,
                               lights=light_sources,
                               scene=scene_settings)

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
                scene: SceneSettings):
    """
    Performs ray tracing for a given set of initial rays, calculating interactions with objects,
    and computing both reflected and go-through ray directions.
    This function is designed to handle the interaction of rays with infinite planes, spheres and cubes.

    :param rays_sources: matrix of ray source coordinates.
    :param rays_directions: matrix of ray direction coordinates.
    :param surfaces: a list of 3d objects. used to fetch normals and calculate reflections.
    :param materials: a list of object materials.
    :param lights: a list of light sources in the scene.
    :param scene: a `SceneSettings` object containing settings for the scene, such as lighting, camera position, etc.
    :return:a 3D array representing the image result
    """
    print(scene.max_recursions)
    recursion_scene = SceneSettings(scene.background_color, scene.root_number_shadow_rays, scene.max_recursions - 1)

    # 6.3: Find the nearest intersection of the ray. This is the surface that will be seen in the image.
    rays_interactions, index_list = compute_rays_interactions(surfaces=surfaces,
                                                              rays_sources=rays_sources,
                                                              rays_directions=rays_directions)

    ray_hits, surfaces_indices = compute_rays_hits(ray_interactions=rays_interactions, index_list=index_list)

    surfaces_normals = get_surfaces_normals(surfaces=surfaces,
                                            surfaces_indices=surfaces_indices,
                                            ray_hits=ray_hits)

    material_indices = get_surfaces_material_indies(surfaces=surfaces, surfaces_indices=surfaces_indices)
    material_colors = get_materials_base_colors(materials=materials, material_indices=material_indices)
    diffusive_colors, base_specular_colors, phong, reflective_colors, transparency_values = material_colors

    # 6.4.1 Go over each light in the scene.
    surfaces_to_lights_directions = compute_light_rays(ray_hits, lights)
    light_color, light_specular_intensity = get_light_base_colors(lights=lights,
                                                                  light_directions=surfaces_to_lights_directions,
                                                                  surfaces=surfaces,
                                                                  hits=ray_hits)

    # 6.4.2: Add the value it induces on the surface.
    specular_colors = compute_specular_colors(surfaces_specular_color=base_specular_colors,
                                              surfaces_phong_coefficient=phong,
                                              surfaces_to_lights_directions=surfaces_to_lights_directions,
                                              viewer_directions=-rays_directions,
                                              surface_normals=surfaces_normals,
                                              light_specular_intensity=light_specular_intensity)

    # transparency_values = np.zeros_like(ray_hits)
    non_transparency_values = np.ones_like(ray_hits) - transparency_values

    base_colors = (diffusive_colors + specular_colors) * light_color * non_transparency_values

    go_through_rays_directions = rays_directions
    if scene.max_recursions > 0:
        go_through_colors = ray_tracing(rays_sources=ray_hits + EPSILON * go_through_rays_directions,
                                        rays_directions=go_through_rays_directions, surfaces=surfaces,
                                        materials=materials, lights=lights, scene=recursion_scene,camera=camera)
    else:
        go_through_colors = np.zeros_like(diffusive_colors)

    back_colors = go_through_colors * transparency_values

    reflection_rays_directions = compute_reflection_rays(-rays_directions, surfaces_normals)
    if scene.max_recursions > 0:
        reflection = ray_tracing(rays_sources=ray_hits + EPSILON * reflection_rays_directions,
                                 rays_directions=reflection_rays_directions, surfaces=surfaces, materials=materials,
                                 lights=lights, scene=recursion_scene,camera=camera)
    else:
        reflection = np.zeros_like(diffusive_colors)
    reflection *= reflective_colors

    image_colors = (back_colors + base_colors + reflection)
    image_colors[image_colors > 1] = 1
    # image_colors = image_colors/np.max(image_colors)

    return image_colors


if __name__ == '__main__':
    main()
