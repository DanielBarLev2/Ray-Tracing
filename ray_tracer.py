from Light import get_light_base_colors, compute_light_rays, compute_specular_colors, compute_diffuse_color
from surfaces.SurfaceAbs import SurfaceAbs, get_surfaces_normals, get_surfaces_material_indices
from ray_functions import get_initial_rays, compute_reflection_rays
from Material import Material, get_materials_base_colors
from Parser import parse_args, parse_scene_file
from SceneSettings import SceneSettings
from BSPNode import BSPNode
from Camera import Camera
from Light import Light
from util import *
import cProfile
import pstats
import re


def main():
    # Parse command line arguments
    args = parse_args()
    # Parse the scene file
    camera, scene_settings, objects, surfaces, materials, light_sources = parse_scene_file(args.scene_file)

    bsp_tree = BSPNode.build_bsp_tree(surfaces=surfaces)

    # 6.1: Discover the location of the pixel on the camera’s screen, Construct a ray from the camera through that pixel
    rays_sources, rays_directions = get_initial_rays(camera, image_width=args.width, image_height=args.height)

    # 6.2: Check the intersection of the ray with all surfaces in the scene
    image_colors = ray_tracing(rays_sources=rays_sources, rays_directions=rays_directions, surfaces=surfaces,
                               materials=materials, lights=light_sources, scene=scene_settings, camera=camera,
                               bsp_tree=bsp_tree).clip(0, 1)

    # Save the output image
    save_image(image_array=image_colors, path=args.output_image, height=args.height, width=args.width)


def ray_tracing(rays_sources: np.ndarray,
                rays_directions: np.ndarray,
                surfaces: list[SurfaceAbs],
                materials: list[Material],
                lights: list[Light],
                scene: SceneSettings, camera: Camera, bsp_tree: BSPNode):
    """
    Performs ray tracing to compute the color of each pixel in the image by simulating ray-object interactions.

    This function handles ray interactions with surfaces including infinite planes, spheres, and cubes.
    It traces rays through the scene, calculates intersections with objects, and computes color contributions
    from diffuse and specular lighting, reflection, and transparency.

    :param rays_sources: ndarray of shape (N, 3) representing the source coordinates of N rays.
    :param rays_directions: ndarray of shape (N, 3) representing the direction vectors of N rays.
    :param surfaces: List of 3D objects in the scene that rays may intersect with.
                     Used to fetch surface normals and calculate reflections.
    :param materials: List of materials associated with the surfaces. Used for determining color properties.
    :param lights: List of light sources in the scene, influencing lighting and shading calculations.
    :param scene: SceneSettings object containing configuration for the scene, such as background color, shadow ray
                  settings, and recursion depth.
    :param camera: Camera object defining the viewpoint of the scene.
    :param bsp_tree: BSPNode object representing a spatial partitioning tree for efficient intersection tests.

    :return: A 2D ndarray of shape (H * W, 3) representing the image, with pixel colors computed from ray tracing.
    """
    if scene.max_recursions < 0:
        return np.full_like(rays_sources, scene.background_color)

    image_colors = np.full_like(rays_sources, scene.background_color)
    recursion_scene = SceneSettings(scene.background_color, scene.root_number_shadow_rays, scene.max_recursions - 1)

    # 6.2: Check the intersection of the ray with all surfaces in the scene
    # 6.3: Find the nearest intersection of the ray. This is the surface that will be seen in the image
    ray_hits, surfaces_indices = bsp_tree.intersect_vectorize(ray_sources=rays_sources, ray_directions=rays_directions)

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
                                                                       hits=ray_hits,
                                                                       shadow_rays_count=scene.root_number_shadow_rays,
                                                                       bsp_tree=bsp_tree)

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
    # output_color := (background_color * transparency) + (diffuse + specular)*(~transparency) + reflection_color
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
                                                          camera=camera, bsp_tree=bsp_tree)
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
                                                  lights=lights, scene=recursion_scene, camera=camera,
                                                  bsp_tree=bsp_tree)
    reflection *= reflective_colors

    image_colors[~bg_pixels] = (back_colors + base_colors + reflection)
    return image_colors


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('time')

    project_path = r'C:\Tau Software\Ray-Tracing'
    escaped_project_path = re.escape(project_path)

    stats.print_stats(escaped_project_path)
