import random

import numpy as np
from camera import Unit_Vector
from ray_tracer import Ray
import intersection_functions

EPSILON = 10 ** -8


def calc_diffuse_reflection(light, light_intensity, surface_diffuse_color, intersection_point, surface_normal):
    L = Unit_Vector(light.position - intersection_point)  # the opposite light dir
    N_dot_L = np.dot(L, surface_normal)
    if N_dot_L < 0:
        # light is behind the surface
        return np.zeros(3, dtype=float)
    return N_dot_L * light_intensity * np.multiply(light.color, surface_diffuse_color)  # element-wise multiplication


def calc_specular_reflection(light, light_intensity, camera, intersection_point,
                             shininess, surface_normal):
    L = Unit_Vector(light.position - intersection_point)  # the opposite light dir
    V = Unit_Vector(camera.position - intersection_point)
    R = L - (2 * np.dot(L, surface_normal) * surface_normal)
    R = Unit_Vector(R)
    R_dot_V = np.dot(R, V)
    if R_dot_V < 0:
        return np.zeros(3, dtype=float)
    return light.color * light_intensity * (R_dot_V ** shininess) * light.specular_intensity


def calc_light_intensity(scene_settings, light_source, intersection_point, surface, objects):
    # Step 1 in pdf: 'Find a plane which is perpendicular to the ray'
    light_ray = Unit_Vector(intersection_point - light_source)
    horizontal_unit = Unit_Vector(light_ray.perpendicular_vector())
    vertical_unit = Unit_Vector(np.cross(light_ray, horizontal_unit))

    # Step 2 in pdf: 'Create the desired square'
    # Start from light position(square center) and go left
    left_bottom_pixel = light_source.position - (light_source.radius / 2) * horizontal_unit.direction
    # Go down
    left_bottom_pixel = left_bottom_pixel - (light_source.radius / 2) * vertical_unit.direction

    # Step 3 in pdf: Scale the RxR square to NxN
    scale_factor = light_source.radius / scene_settings.root_number_shadow_rays
    # new scaled vectors for NxN square
    horizontal = horizontal_unit.direction * scale_factor
    vertical = vertical_unit.direction * scale_factor

    # Step 4 in pdf: Shoot rays from cells to intersection point
    hit_points_counter = 0
    for i in range(scene_settings.root_number_shadow_rays):
        for j in range(scene_settings.root_number_shadow_rays):
            random_cell_point = left_bottom_pixel + (i + random.random()) * horizontal + (
                        j + random.random()) * vertical
            ray = Ray(random_cell_point, intersection_point)
            intersected_surfaces = intersection_functions.find_intersections(ray, objects)

            #  BONUS : for each object that is between the cell and the intersection point
            #           we will multiply it's transparency with the previous objects' transparency
            #           the result: a floating point number between 0 to 1, that will replace the binary value of
            #           intersecting or not intersecting
            light_result = 1  # the "amount" of light that reached the intersection point from the cell
            for item in intersected_surfaces:
                material = objects[item[0].material_index]
                light_result *= material.transparency
                t = item[1]
                item_intersection = ray.camera_pos + ray.ray_direction * t  # camera_pos = random_cell_point
                if np.linalg.norm(item_intersection, intersection_point) < EPSILON:
                    # The ray intersected the desired surface, need to stop
                    hit_points_counter += light_result
                    break

    hit_precentage = hit_points_counter / (scene_settings.root_number_shadow_rays**2)
    return (1 - light_source.shadow_intensity) + (light_source.shadow_intensity * hit_precentage)
