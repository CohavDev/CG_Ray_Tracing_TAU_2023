import math
import numpy as np
from camera import Unit_Vector


def calc_diffuse_reflection(light, light_intensity, surface_diffuse_color, intersection_point, surface_normal):
    L = Unit_Vector(light.position - intersection_point)  # the opposite light dir
    N_dot_L = np.dot(L, surface_normal)
    if N_dot_L < 0:
        # light is behind the surface
        return np.zeros(3, dtype=float)
    return N_dot_L * light_intensity * np.multiply(light.color, surface_diffuse_color)  # element-wise multiplication


def calc_specular_reflection(light, light_intensity, surface_specular_color, camera, intersection_point,
                             shininess, surface_normal):
    L = Unit_Vector(light.position - intersection_point)  # the opposite light dir
    V = Unit_Vector(camera.position - intersection_point)
    R = L - (2 * np.dot(L, surface_normal) * surface_normal)
    R = Unit_Vector(R)
    R_dot_V = np.dot(R, V)
    if R_dot_V < 0:
        return np.zeros(3, dtype=float)
    return light.color * light_intensity * (R_dot_V ** shininess) * light.specular_intensity
