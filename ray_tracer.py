import argparse
import numpy as np
from PIL import Image
from Light import Light
from Camera import Camera
from Material import Material
from surfaces.Cube import Cube
from surfaces.Sphere import Sphere
from SceneSettings import SceneSettings
from surfaces.InfinitePlane import InfinitePlane


def parse_scene_file(file_path):
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
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects_3D.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects_3D.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
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

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    # 6.1:
    view_matrix = camera.create_view_matrix()


    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array=image_array, path=args.output_image)



def shoot_pixels_rays(camera: Camera, image_width:int, image_height:int):
    X_DIRECTION = np.array([1,0,0])
    Y_DIRECTION = np.array([0,1,0])
    Z_DIRECTION = np.array([0,0,1])

    w = camera.screen_width
    h = w / image_width * image_height

    h_granularity = h/image_height;
    w_granularity = w/image_width;
    screen_center = camera.position + Z_DIRECTION* camera.screen_distance
    pixel_0_0 = screen_center + (h/2 * Y_DIRECTION) + (h/2 * X_DIRECTION)
    # todo: change to matrix calculation
    for i in range(image_width):
        for j in range(image_height):
            ray = pixel_0_0 - (i * Y_DIRECTION) -(j * X_DIRECTION)


if __name__ == '__main__':
    main()
