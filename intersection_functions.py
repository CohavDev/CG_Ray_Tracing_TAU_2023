import math
import numpy as np

from surfaces import cube
from surfaces import infinite_plane
from surfaces import sphere as Sphere

EPSILON = 10**-10


def plane_intersection(ray, inf_plane):
    if np.dot(ray.ray_direction.direction, np.array(inf_plane.normal)) == 0:
        # the ray does not intersect the plane (ray is parallel to the plane)
        return -1
    # else
    # t is the amount of ray.dir needed to get from p0(camera) to the intersection point on the plane
    t = -1 * (np.dot(ray.camera_pos, inf_plane.normal) - inf_plane.offset) / np.dot(ray.ray_direction.direction, inf_plane.normal)
    return t


def sphere_intersection(ray, sphere):
    # according to the geometric method in the presentation
    L = np.subtract(sphere.position, ray.camera_pos)
    t_ca = np.dot(L, ray.ray_direction.direction)
    if t_ca < 0:
        return -1
    d_2 = np.dot(L, L) - pow(t_ca, 2)
    r_2 = pow(sphere.radius, 2)
    if d_2 > r_2:
        return -1
    t_hc = math.sqrt(r_2 - d_2)
    t_close = t_ca - t_hc  # close point of intersection
    t_far = t_ca + t_hc  # far point of intersection
    return t_close


def cube_intersection(ray, cube):
    # getting min/max of all coordinates
    slab_xmax = cube.position[0] + cube.scale / 2
    slab_xmin = cube.position[0] - cube.scale / 2

    slab_ymax = cube.position[1] + cube.scale / 2
    slab_ymin = cube.position[1] - cube.scale / 2

    slab_zmax = cube.position[2] + cube.scale / 2
    slab_zmin = cube.position[2] - cube.scale / 2

    # calculating tmin/tmax for each coordinate:
    ray_direction_temp = ray.ray_direction.direction[0]
    if ray.ray_direction.direction[0] == 0:
        ray_direction_temp = EPSILON
    t_xmin = (slab_xmin - ray.camera_pos[0]) / ray_direction_temp
    t_xmax = (slab_xmax - ray.camera_pos[0]) / ray_direction_temp

    if t_xmin > t_xmax:
        # Replace the two, in case the ray is from the 'positive' side of x-axis:
        temp = t_xmin
        t_xmin = t_xmax
        t_xmax = temp

    ray_direction_temp = ray.ray_direction.direction[1]
    if ray.ray_direction.direction[1] == 0:
        ray_direction_temp = EPSILON
    t_ymin = (slab_ymin - ray.camera_pos[1]) / ray_direction_temp
    t_ymax = (slab_ymax - ray.camera_pos[1]) / ray_direction_temp

    if t_ymin > t_ymax:
        # Replace the two, in case the ray is from the 'positive' side of y-axis:
        temp = t_ymin
        t_ymin = t_ymax
        t_ymax = temp

    if t_xmin > t_ymax or t_ymin > t_xmax:
        # the ray meets one minimum coordinate that exceeds another max coordinate
        # therefore does not intersects the cube
        return -1

    # Reducing t to tx check further intersections
    if t_ymin > t_xmin:
        t_xmin = t_ymin
    if t_ymax < t_xmax:
        t_xmax = t_ymax

    ray_direction_temp = ray.ray_direction.direction[2]
    if ray.ray_direction.direction[2] == 0:
        ray_direction_temp = EPSILON
    t_zmin = (slab_zmin - ray.camera_pos[2]) / ray_direction_temp
    t_zmax = (slab_zmax - ray.camera_pos[2]) / ray_direction_temp

    if t_zmin > t_zmax:
        # Replace the two, in case the ray is from the 'positive' side of z-axis:
        temp = t_zmin
        t_zmin = t_zmax
        t_zmax = temp

    # the ray meets one minimum coordinate that exceeds another max coordinate
    # therefore does not intersects the cube
    if t_xmin > t_zmax or t_zmin > t_xmax:
        return -1

    # final t is in t_xmin
    if t_zmin > t_xmin:
        t_xmin = t_zmin
    return t_xmin


def find_intersections(ray, objects):
    surfaces_list = []
    for surface in objects:
        t = 0  # init
        if isinstance(surface, cube.Cube):
            t = cube_intersection(ray, surface)
        elif isinstance(surface, Sphere.Sphere):
            t = sphere_intersection(ray, surface)
        elif isinstance(surface, infinite_plane.InfinitePlane):
            t = plane_intersection(ray, surface)
        if t > EPSILON:
            surfaces_list.append((surface, t))
    surfaces_list.sort(key=lambda tup: tup[1])  # sort by t (distance)
    return surfaces_list
