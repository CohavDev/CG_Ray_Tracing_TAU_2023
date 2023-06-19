import numpy as np


class Unit_Vector:

    def __init__(self, direction):
        self.direction = direction / np.linalg.norm(direction)  # normalized vector

    def perpendicular_vector(self):
        # Find a perpendicular vector to self, with respect to x-axis or y-axis
        x_unit_vector = np.array([1, 0, 0])
        vector = np.cross(self.direction, x_unit_vector)
        if np.all((vector == 0)):
            y_unit_vector = np.array([0, 1, 0])
            vector = Unit_Vector(np.cross(self.direction, np.array(y_unit_vector))).direction
        return vector


class Screen:
    #  describes the screen of the camera
    def __init__(self, left_bottom_pixel_center, horizontal_scaled, vertical_scaled):
        self.left_bottom_pixel_center = left_bottom_pixel_center
        self.horizontal_scaled = horizontal_scaled  # size of step required to move to the next horizontal pixel
        self.vertical_scaled = vertical_scaled  # likewise for vertical


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = position
        self.look_at = look_at
        self.up_vector = Unit_Vector(up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.towards = Unit_Vector(np.subtract(look_at, position))

    def fix_up_vector(self):
        horizontal = Unit_Vector(np.cross(self.towards.direction, self.up_vector.direction))
        # getting real up direction vector:
        vertical = Unit_Vector(np.cross(self.towards.direction, horizontal.direction))

        # Fix up vector:
        self.up_vector = vertical.direction
        return horizontal, vertical


def build_screen(camera, image_pixel_width, image_pixel_height):
    horizontal, vertical = camera.fix_up_vector()

    # Gets the screen's center point:
    screen_center = camera.position + camera.towards.direction * camera.screen_distance

    # Calculating screen height according to image ratio
    screen_height = camera.screen_width * (image_pixel_height / image_pixel_width)

    # go to the left bottom corner using scaled horizontal & vertical vectors
    # (from the prespective of the camera position)
    left_bottom_pixel = screen_center + (
                camera.screen_width / 2 * horizontal.direction)  # move to the left screen's edge
    left_bottom_pixel -= (screen_height / 2 * vertical.direction)  # move to the bottom of the screen

    # How many coordinates to move in the screen to get to the next pixel
    pixel_width = camera.screen_width / image_pixel_width
    pixel_height = screen_height / image_pixel_height
    horizontal.direction = horizontal.direction * (-1)  # Make horizontal vector point to screen's right side

    # Creating the screen's scaled vectors
    # (vectors are from camera perspective)
    screen_horizontal_pixel_step_vector = Unit_Vector(horizontal.direction)  # Unit vector in the right dir of screen

    # Scaling the vector to the size of 1 screen's pixel:
    screen_horizontal_pixel_step_vector.direction = pixel_width * horizontal.direction
    screen_vertical_pixel_step_vector = Unit_Vector(vertical.direction)
    screen_vertical_pixel_step_vector.direction = pixel_height * vertical.direction

    # Sets left-bottom pixel to be the center of that pixel:
    left_bottom_pixel += 0.5 * screen_horizontal_pixel_step_vector.direction + 0.5 * screen_vertical_pixel_step_vector.direction

    # Create screen object:
    screen = Screen(left_bottom_pixel, screen_horizontal_pixel_step_vector.direction,
                    screen_vertical_pixel_step_vector.direction)
    return screen
