from surfaces.InfinitePlane import InfinitePlane
from surfaces.SurfaceAbs import SurfaceAbs
from surfaces.Background import Background
from surfaces.Sphere import Sphere
from surfaces.Cube import Cube

from SceneSettings import SceneSettings
from Material import Material
from Camera import Camera
from Light import Light

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()
    return args


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
