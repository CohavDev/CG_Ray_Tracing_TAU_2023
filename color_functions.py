import random

import numpy as np
from camera import Unit_Vector
from ray_tracer import Ray, get_surface_normal
import intersection_functions
from surfaces import infinite_plane

EPSILON = 10 ** -9
c = 0


def calc_diffuse_reflection(light, light_intensity, surface_diffuse_color, intersection_point, surface_normal):
    L = Unit_Vector(light.position - intersection_point)  # the opposite light dir
    N_dot_L = np.dot(L.direction, surface_normal.direction)
    if N_dot_L < 0:
        # light is behind the surface
        return np.zeros(3, dtype=float)
    return N_dot_L * light_intensity * np.multiply(light.color, surface_diffuse_color)  # element-wise multiplication


def calc_specular_reflection(light, light_intensity, origin_position, intersection_point,
                             shininess, surface_normal):
    L = Unit_Vector(light.position - intersection_point).direction  # the opposite light dir
    V = Unit_Vector(origin_position - intersection_point)
    R = L - (2 * np.dot(L, surface_normal.direction) * surface_normal.direction)
    R = Unit_Vector(R)
    R_dot_V = np.dot(R.direction, V.direction)
    if R_dot_V < 0:
        return np.zeros(3, dtype=float)
    return np.array(light.color) * light_intensity * (R_dot_V ** shininess) * light.specular_intensity


def calc_light_intensity(scene_settings, light_source, intersection_point, surface, objects):
    # Step 1 in pdf: 'Find a plane which is perpendicular to the ray'
    light_ray = Unit_Vector(intersection_point - light_source.position)
    horizontal_unit = Unit_Vector(light_ray.perpendicular_vector())
    vertical_unit = Unit_Vector(np.cross(light_ray.direction, horizontal_unit.direction))

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
    for i in range(int(scene_settings.root_number_shadow_rays)):
        for j in range(int(scene_settings.root_number_shadow_rays)):
            random_cell_point = left_bottom_pixel + (i + random.random()) * horizontal + (
                    j + random.random()) * vertical
            ray = Ray(random_cell_point, intersection_point)
            intersected_surfaces = intersection_functions.find_intersections(ray, objects)

            #  BONUS : for each object that is between the cell and the intersection point
            #           we will multiply it's transparency with the previous objects' transparency
            #           the result: a floating point number between 0 to 1, that will replace the binary value of
            #           intersecting or not intersecting
            light_result = 1  # the "amount" of light that reached the intersection point from the cell
            if np.linalg.norm(np.subtract(
                    ray.camera_pos + ray.ray_direction.direction * intersected_surfaces[0][1], intersection_point)) < EPSILON:
                hit_points_counter += 1
            # for item in intersected_surfaces:
            #     material = objects[item[0].material_index]
            #     light_result *= material.transparency
            #     t = item[1]
            #     item_intersection = ray.camera_pos + ray.ray_direction.direction * t  # camera_pos = random_cell_point
            #     if np.linalg.norm(np.subtract(item_intersection, intersection_point)) < EPSILON:
            #         # The ray intersected the desired surface, need to stop
            #         hit_points_counter += light_result
            #         break

    hit_precentage = hit_points_counter / (scene_settings.root_number_shadow_rays ** 2)
    return (1 - light_source.shadow_intensity) + (light_source.shadow_intensity * hit_precentage)


def get_surface_color(intersected_surfaces, current_surface_index, objects, lights, origin_position, ray,
                      scene_settings,
                      current_recursion_depth):
    surface = intersected_surfaces[current_surface_index][0]
    t = intersected_surfaces[current_surface_index][1]
    material = objects[surface.material_index]  # The material of the current surface
    diffuse_color = np.zeros(3, dtype=float)
    specular_color = np.zeros(3, dtype=float)
    background_color = np.array(scene_settings.background_color)

    # Getting the intersection point of the ray and the surface:
    intersection_point = ray.get_point_from_t(t)
    # Getting the normal to the surface in the intersection-point:
    surface_normal = get_surface_normal(surface, intersection_point)

    # Loop through all lights in the scene and calculate their affect on the intersection point
    for light in lights:
        # Calculating light intensity and adding the diffuse and specular color to the total, of each light
        light_intensity = calc_light_intensity(scene_settings, light, intersection_point, surface, objects)
        diffuse_color += calc_diffuse_reflection(light, light_intensity, material.diffuse_color,
                                                 intersection_point, surface_normal)
        specular_color += calc_specular_reflection(light, light_intensity, origin_position, intersection_point,
                                                   material.shininess, surface_normal)
    # if isinstance(surface, infinite_plane.InfinitePlane):
    #     print("diffuse color of plane: ", diffuse_color)
    diffuse_color = material.diffuse_color * diffuse_color
    specular_color = material.specular_color * specular_color

    # Handle Reflection : creating the reflection vector and recursively ray-casting
    reflection_vec = ray.ray_direction.direction - 2 * np.dot(ray.ray_direction.direction, surface_normal.direction) * \
                     surface_normal.direction
    reflection_vec = Unit_Vector(reflection_vec)
    reflection_ray = Ray(intersection_point, intersection_point + reflection_vec.direction)
    reflection_color = get_pixel_color(intersection_point, reflection_ray, scene_settings, objects, lights,
                                       current_recursion_depth + 1)
    reflection_color = material.reflection_color * reflection_color

    # Handle the case where the ray intersected transparent material and there are more surfaces 'after' the surface
    if current_surface_index < len(intersected_surfaces) - 1:
        if material.transparency != 0:
            # Pass the ray through
            background_color *= get_surface_color(intersected_surfaces, current_surface_index + 1, objects, lights,
                                                  origin_position, ray, scene_settings,
                                                  0)  # TODO: 0?
    output_color = background_color * material.transparency + \
                   (diffuse_color + specular_color) * (1 - material.transparency) + reflection_color
    return output_color


def get_pixel_color(origin_position, ray, scene_settings, objects, lights, current_recursion_depth):
    global c
    max_recursion = scene_settings.max_recursions
    if current_recursion_depth >= max_recursion:
        # return bg color
        return np.array(scene_settings.background_color)
    collide_objects = intersection_functions.find_intersections(ray, objects)
    if len(collide_objects) == 0:
        # return bg color, no surfaces intersected
        return np.array(scene_settings.background_color)
    c += 1
    # Call get_surface_color that is defined above
    return np.array(get_surface_color(collide_objects, 0, objects, lights, origin_position, ray, scene_settings,
                                      current_recursion_depth))
