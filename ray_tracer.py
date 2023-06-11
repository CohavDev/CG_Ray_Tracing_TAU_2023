import argparse
from PIL import Image
import numpy as np

from camera import *
from intersection_functions import *
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


class Ray:
    def __init__(self, camera_pos, pixel_center):
        self.camera_pos = camera_pos
        self.pixel_center = pixel_center
        self.ray_direction = Unit_Vector(
            np.subtract(pixel_center, camera_pos))  # Vector from camera to the pixel's center

    def get_point_from_t(self, t):
        return self.camera_pos + t * np.array(self.ray_direction)


def send_ray_through_pixel(camera, screen, image_i, image_j):
    # getting the pixel in the screen corresponding to the pixel (i,j) in image:
    pix_center = screen.left_bottom_pixel_center + image_i * screen.horizontal_scaled + image_j * screen.vertical_scaled
    # Ray from camera to the pixel corresponding to pixel (i,j) in image:
    ray = Ray(camera.position, pix_center)
    return ray


def get_pixel_color(camera, ray, scene_settings, objects, current_recursion_depth):
    max_recursion = scene_settings.max_recursions
    if current_recursion_depth >= max_recursion:
        # return bg color
        return np.array(scene_settings.background_color)
    collide_objects = find_intersections(ray, objects)  # TODO: change later
    if len(collide_objects) == 0:
        # return bg color
        return np.array(scene_settings.background_color)


def get_surface_color(surface, objects, camera, ray, scene_settings, current_recursion_depth):
    #  TODO:need to complete later
    return


def get_cube_normal(cube, intersection_point):
    x = cube.position[0]
    y = cube.position[1]
    z = cube.position[2]
    EPSILON = 10 ** -8
    # because cube is parallel to each axis, we can determine on which side of the cube the point is on
    # intersects positive x side of the cube
    if abs((intersection_point[0] - x) - cube.scale / 2) < EPSILON:
        return np.array((1, 0, 0))
    # intersects negative x side of the cube
    elif abs((x - intersection_point[0]) - cube.scale / 2) < EPSILON:
        return np.array((-1, 0, 0))
    # intersects positive y side of the cube
    elif abs((intersection_point[1] - y) - cube.scale / 2) < EPSILON:
        return np.array((0, 1, 0))
    # intersects negative y side of the cube
    elif abs((y - intersection_point[1]) - cube.scale / 2) < EPSILON:
        return np.array((0, -1, 0))
    # intersects positive z side of the cube
    elif abs((intersection_point[2] - z) - cube.scale / 2) < EPSILON:
        return np.array((0, 0, 1))
    # intersects negative z side of the cube
    else:
        return np.array((0, 0, -1))


def get_surface_normal(surface, intersection_point):
    if isinstance(surface, Sphere.Sphere):
        return Unit_Vector(intersection_point - surface.position)

    elif isinstance(surface, infinite_plane.InfinitePlane):
        return surface.normal

    else:
        return get_cube_normal(surface, intersection_point)


def parse_scene_file(file_path):
    objects = []
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
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
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
    image_height = args.height
    image_width = args.width
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    screen = build_screen(camera, image_width, image_height)  # TODO:args names might be incorrect
    # Initialize an empty image
    image_array = np.zeros((image_height, image_width, 3))
    for i in range(image_height):
        for j in range(image_width):
            ray = send_ray_through_pixel(camera, screen, j, i)  # sending ray through each image's pixel
            # calculate color for pixel
            color = get_pixel_color(camera, ray, scene_settings, objects, 0)
            # set color for pixel
            image_array[i][j] = color

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
