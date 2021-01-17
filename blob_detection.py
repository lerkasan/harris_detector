import os
import math
import random
import numpy as np
import correlation as corr
from PIL import Image
from enum import Enum

INPUT_DIR = 'files/1/'
OUTPUT_DIR = 'files/1/'

# Parameters for blurring with Gaussian filter
# Default GAUSSIAN_KERNEL_HEIGHT and GAUSSIAN_KERNEL_WIDTH are 23
GAUSSIAN_KERNEL_HEIGHT = 21
GAUSSIAN_KERNEL_WIDTH = 21
# Default GAUSSIAN_SIGMA is 7
GAUSSIAN_SIGMA = 5

# A window size for calculating covariance matrix. Default value is 3
WINDOW_SIZE = 3

# HARRIS_DETECTOR_ALPHA should be in a range of [0.04; 0.06]
HARRIS_DETECTOR_ALPHA = 0.04
# HARRIS_DETECTOR_THRESHOLD should be >= 0. Default value 1 * math.pow(10, -11). Value 2.5 * math.pow(10, -13) is also good
HARRIS_DETECTOR_THRESHOLD = 1 * math.pow(10, -11)

# Default DISTANCE_THRESHOLD = 78
DISTANCE_THRESHOLD = 78
MIN_NEIGHBORS = 4


class DerivativeOperator(list, Enum):
    LAPLACIAN = [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
    SOBEL = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]


class Status(str, Enum):
    VISITED = 'Visited',
    NOISE = 'Noise'


def get_median_filter(size: int) -> list:
    if size <= 0 or size % 2 == 0:
        raise ValueError('Size of a median filter should positive odd number.')
    median_filter = np.ones((size, size), 'uint8') / (size ** 2)
    return median_filter


def gaussian_distribution(x: float, y: float, sigma: float) -> float:
    if sigma == 0:
        raise ValueError('Sigma should not be zero otherwise division by zero will occur.')
    power = -((x ** 2) + (y ** 2))/(2 * (sigma ** 2))
    gaussian = math.exp(power) / (2 * math.pi * (sigma ** 2))
    return gaussian


def get_gaussian_grayscale_kernel(kernel_rows_number: int, kernel_columns_number: int, sigma: float) -> np.ndarray:
    if kernel_rows_number <= 0 or kernel_columns_number <= 0:
        raise ValueError('kernel_rows_number and kernel_columns_number parameters should be positive integers.')
    kernel_row_offset = kernel_rows_number // 2
    kernel_column_offset = kernel_columns_number // 2
    kernel_row_range_max = kernel_row_offset + kernel_rows_number % 2
    kernel_column_range_max = kernel_column_offset + kernel_columns_number % 2
    gaussian_kernel = [[0.0 for y in range(kernel_columns_number)] for x in range(kernel_rows_number)]
    for row in range(-kernel_row_offset, kernel_row_range_max):
        for column in range(-kernel_column_offset, kernel_column_range_max):
            gaussian_kernel[row + kernel_row_offset][column + kernel_column_offset] = gaussian_distribution(row, column, sigma)
    return np.asarray(gaussian_kernel)


def save_gaussian_kernel_as_image(kernel_data, output_path: str):
    kernel_data = np.asarray(kernel_data)
    kernel_rows_number = kernel_data.shape[0]
    kernel_columns_number = kernel_data.shape[1]
    kernel_center_row_index = kernel_rows_number // 2
    kernel_center_column_index = kernel_columns_number // 2
    grayscale_gaussian_kernel = kernel_data / kernel_data[kernel_center_row_index][kernel_center_column_index]
    corr.save_pixels_as_grayscale_image(grayscale_gaussian_kernel, output_path)


def blur_image(image_data, kernel_rows_number: int, kernel_columns_number: int, sigma: float, output_dir: str = None) -> np.ndarray:
    kernel_row_offset = kernel_rows_number // 2
    kernel_column_offset = kernel_columns_number // 2
    padded_image = corr.image_padding(image_data, kernel_row_offset, kernel_column_offset)
    gaussian_kernel = get_gaussian_grayscale_kernel(kernel_rows_number, kernel_columns_number, sigma)
    blurred_image = corr.get_convolution(padded_image, gaussian_kernel)
    if not (output_dir is None):
        corr.save_pixels_as_grayscale_image(blurred_image, output_dir + 'blur.png')
        save_gaussian_kernel_as_image(gaussian_kernel, output_dir + 'kernel.png')
    return blurred_image


def get_derivative(image_data, operator, output_dir: str = None) -> np.ndarray:
    kernel_offset = len(operator) // 2
    padded_image = corr.image_padding(image_data, kernel_offset, kernel_offset)
    image_derivative = corr.get_convolution(padded_image, operator)
    if not (output_dir is None):
        corr.save_pixels_as_grayscale_image(image_derivative, output_dir + 'derivative.png')
    return image_derivative


def get_harris_cornerness(image_data, window_size: int = WINDOW_SIZE, alpha: float = HARRIS_DETECTOR_ALPHA,
                          dx_operator: DerivativeOperator = DerivativeOperator.LAPLACIAN, enable_blur: bool = True) -> list:
    image_rows_number = image_data.shape[0]
    image_columns_number = image_data.shape[1]
    window_size_offset = window_size // 2
    window_size_max_index = window_size // 2 + window_size % 2
    if enable_blur:
        blurred_image = blur_image(image_data, GAUSSIAN_KERNEL_HEIGHT, GAUSSIAN_KERNEL_WIDTH, GAUSSIAN_SIGMA)
    else:
        blurred_image = image_data
    dy_operator = np.transpose(dx_operator)
    image_dx = get_derivative(blurred_image, dx_operator)
    image_dy = get_derivative(blurred_image, dy_operator)
    dx_squared = np.multiply(image_dx, image_dx)
    dy_squared = np.multiply(image_dy, image_dy)
    dx_times_dy = np.multiply(image_dx, image_dy)

    cornerness = [[0.0 for y in range(image_columns_number)] for x in range(image_rows_number)]
    for row in range(image_rows_number):
        for column in range(image_columns_number):
            dx_squared_sum = 0.0
            dy_squared_sum = 0.0
            dx_times_dy_sum = 0.0
            for window_row in range(-window_size_offset, window_size_max_index):
                for window_column in range(-window_size_offset, window_size_max_index):
                    if 0 <= row + window_row < image_rows_number and 0 <= column + window_column < image_columns_number:
                        dx_squared_sum += dx_squared[row + window_row][column + window_column]
                        dy_squared_sum += dy_squared[row + window_row][column + window_column]
                        dx_times_dy_sum += dx_times_dy[row + window_row][column + window_column]
            det = dx_squared_sum * dy_squared_sum - (dx_times_dy_sum ** 2)
            trace = dx_squared_sum + dy_squared_sum
            cornerness[row][column] = det - alpha * (trace ** 2)
    return cornerness


def non_maximum_suppression(cornerness: list, threshold: float, window_radius: int) -> list:
    # Non-maximum supression implementation is based on the article "An Analysis and Implementation of
    # the Harris Corner Detector" by Javier Sanchez, Nelson Monzon, and Agustín Salgado
    corners = []
    cornerness_rows_number = len(cornerness)
    cornerness_columns_number = len(cornerness[0])

    skip = [[False for y in range(cornerness_columns_number)] for x in range(cornerness_rows_number)]

    for row in range(cornerness_rows_number):
        for column in range(cornerness_columns_number):
            if cornerness[row][column] < threshold:
                skip[row][column] = True

    for row in range(window_radius, cornerness_rows_number - window_radius):
        column = window_radius
        while column < (cornerness_columns_number - window_radius) and \
                (skip[row][column] or (cornerness[row][column - 1] >= cornerness[row][column])):
            column += 1
        while column < (cornerness_columns_number - window_radius):
            while column < (cornerness_columns_number - window_radius) and \
                    (skip[row][column] or (cornerness[row][column + 1] >= cornerness[row][column])):
                column += 1
            if column < (cornerness_columns_number - window_radius):
                column1 = column + 2
                while (column1 <= column + window_radius) and (cornerness[row][column1] < cornerness[row][column]):
                    skip[row][column1] = True
                    column1 += 1
                if column1 > column + window_radius:
                    column2 = column - 1
                    while column2 >= (column - window_radius) and (cornerness[row][column2] <= cornerness[row][column]):
                        column2 -= 1
                    if column2 < column - window_radius:
                        row_for_comparison = row + window_radius
                        found = False
                        while (not found) and (row_for_comparison > row):
                            column_for_comparison = column + window_radius
                            while (not found) and (column_for_comparison >= column - window_radius):
                                if cornerness[row_for_comparison][column_for_comparison] > cornerness[row][column]:
                                    found = True
                                else:
                                    skip[row_for_comparison][column_for_comparison] = True
                                column_for_comparison -= 1
                            row_for_comparison -= 1
                        row_for_comparison = row - window_radius
                        while (not found) and (row_for_comparison < row):
                            column_for_comparison = column - window_radius
                            while (not found) and (column_for_comparison <= column + window_radius):
                                if cornerness[row_for_comparison][column_for_comparison] >= cornerness[row][column]:
                                    found = True
                                column_for_comparison += 1
                            row_for_comparison += 1
                        if not found:
                            # corners.append((row, column, cornerness[row][column]))
                            corners.append((row, column))
                column = column1
    return corners


def convert_grayscale_to_rgb(image_data) -> np.ndarray:
    image_data = np.asarray(image_data)
    if len(image_data.shape) == 3:
        return image_data.astype(np.uint8)
    if len(image_data.shape) == 2:
        image_rows_number = image_data.shape[0]
        image_columns_number = image_data.shape[1]
        image_rgb_data = np.zeros((image_rows_number, image_columns_number, 3), 'uint8')
        for row in range(image_rows_number):
            for column in range(image_columns_number):
                image_rgb_data[row][column] = [image_data[row][column], image_data[row][column], image_data[row][column]]
        return np.asarray(image_rgb_data).astype(np.uint8)


def mark_corners(image_data, corners, rgb_color: list = (0, 255, 0)) -> np.ndarray:
    mark_radius = 1
    image_rgb_data = convert_grayscale_to_rgb(image_data)
    image_rows_number = image_rgb_data.shape[0]
    image_columns_number = image_rgb_data.shape[1]
    for corner in corners:
        corner_row = corner[0]
        corner_column = corner[1]
        for mark_row in range(-mark_radius, mark_radius):
            for mark_column in range(-mark_radius, mark_radius):
                if 0 <= corner_row + mark_row < image_rows_number and 0 <= corner_column + mark_column < image_columns_number:
                    image_rgb_data[corner_row + mark_row][corner_column + mark_column] = rgb_color
    return image_rgb_data


def harris_corner_detector(image_data, window_size: int, alpha: float, threshold: float, output_path: str, show_corners: bool = False) -> list:
    cornerness = get_harris_cornerness(image_data, window_size, alpha, dx_operator=DerivativeOperator.LAPLACIAN, enable_blur=True)
    corners = non_maximum_suppression(cornerness, threshold, window_size * 2)
    outer_corners = get_outer_corners(corners)
    image_rgb_data = corr.denormalize_grayscale_pixels(image_data)
    marked_rgb_data = mark_blob(image_rgb_data, outer_corners)

    clusters = dbscan(corners, DISTANCE_THRESHOLD, MIN_NEIGHBORS)

    if show_corners:
        # marked_rgb_data = mark_corners(marked_rgb_data, corners)
        # image_data = Image.fromarray(marked_rgb_data)
        color = 50
        cluster_corner_coords = []

        clusters_coords = []

        stripped_outputpath = output_path.split('.png', 1)[0]

        for cluster in clusters:
            cluster_corner_coords = []
            for cluster_corner in cluster:
                cluster_corner_coord = cluster_corner[1]
                cluster_corner_coords.append(cluster_corner_coord)
            clusters_coords.append(cluster_corner_coords)

        old_marked_rgb_data = marked_rgb_data
        for cluster in clusters_coords:
            color += 35
            # marked_rgb_data = mark_corners(marked_rgb_data, cluster_corner_coords, [color, color, color])
            marked_rgb_data = mark_corners(old_marked_rgb_data, cluster, [0, 255, 0])
            image_data = Image.fromarray(marked_rgb_data)
            image_data.save(stripped_outputpath + str(color) + '.png')

    # if show_corners:
    #     grouped_corners = find_corner_neighbors(corners, DISTANCE_THRESHOLD)
    #     for group in grouped_corners:
    #         color = random(255)
    #         marked_rgb_data = mark_corners(marked_rgb_data, corners)
    #     image_data = Image.fromarray(marked_rgb_data)

    print(clusters)
    # try:
    #     image_data.save(output_path)
    # except IOError:
    #     print('Cannot save image as a file ', output_path)
    return outer_corners


def get_outer_corners(corners: list) -> list:
    if len(corners) == 0:
        raise ValueError('List of corners is empty due to: '
                         '1) too large threshold in Harris detector  or '
                         '2) too large size of smoothening kernel. Please adjust those parameters.')
    else:
        x_min, y_min = corners[0][0], corners[0][1]
        x_max, y_max = corners[0][0], corners[0][1]
        for corner in corners:
            if corner[0] < x_min:
                x_min = corner[0]
            if corner[1] < y_min:
                y_min = corner[1]
            if corner[0] > x_max:
                x_max = corner[0]
            if corner[1] > y_max:
                y_max = corner[1]
        return [x_min, y_min, x_max, y_max]


# def find_corner_neighbors(corners: list, threshold: float) -> list:
#     corners_number = len(corners)
#     grouped_corners = []
#     distances = np.zeros((corners_number, corners_number), np.float)
#     for corner1_index in range(corners_number):
#         corner_neighbors = []
#         for corner2_index in range(corner1_index + 1, corners_number):
#             distance = get_distance(corners[corner1_index], corners[corner2_index])
#             distances[corner1_index][corner2_index] = distance
#             distances[corner2_index][corner1_index] = distance
#             if distance < threshold:
#                 corner_neighbors.append((corners[corner2_index][0], corners[corner2_index][1]))
#         if corner_neighbors:
#             corner_neighbors.append((corners[corner1_index][0], corners[corner1_index][1]))
#             grouped_corners.append(corner_neighbors)
#     return grouped_corners

def get_corner_neighbors(corner, corners: list, threshold: float) -> set:
    corner_neighbors = set()
    for corner_elem in corners:
        distance = get_distance(corner[1], corner_elem[1])
        if 0 < distance < threshold:
            corner_neighbors.add(corner_elem)
    return corner_neighbors


# def dbscan(corners: list, threshold: float, min_neighbors: int) -> list:
#     corners_number = len(corners)
#     clusters = []
#     statuses = ['' for n in range(corners_number)]
#     for corner_index in range(corners_number):
#         corner = corners[corner_index]
#         if statuses[corner_index] != Status.VISITED:
#             statuses[corner_index] = Status.VISITED
#             neighbors = get_corner_neighbors(corner, corners, threshold)
#             if len(neighbors) < min_neighbors:
#                 statuses[corner_index] = Status.NOISE
#             else:
#                 cluster = set()
#                 while neighbors:
#                     neighbor = neighbors.pop()
#                     if statuses[neighbor] != Status.VISITED:
#                         statuses[neighbor] = Status.VISITED
#                         extended_neighbors = get_corner_neighbors(neighbor)
#                         if len(neighbors) >= min_neighbors:
#                             neighbors.update(extended_neighbors)
#                     # if ():
#                         cluster.add(neighbor)
#                 clusters.append(cluster)
#     return clusters

#
def dbscan(corners: list, threshold: float, min_neighbors: int) -> list:
    corners_number = len(corners)
    corners_with_index = []
    for corner_index in range(corners_number):
        corners_with_index.append((corner_index, (corners[corner_index])))
    clusters = []
    statuses = ['' for n in range(corners_number)]
    for corner_index in range(corners_number):
        corner = corners_with_index[corner_index]
        if statuses[corner_index] != Status.VISITED:
            statuses[corner_index] = Status.VISITED
            neighbors = get_corner_neighbors(corner, corners_with_index, threshold)
            if len(neighbors) < min_neighbors:
                statuses[corner_index] = Status.NOISE
            else:
                cluster = set()
                while neighbors:
                    neighbor = neighbors.pop()
                    neighbor_index = neighbor[0]
                    if statuses[neighbor_index] != Status.VISITED:
                        statuses[neighbor_index] = Status.VISITED
                        extended_neighbors = get_corner_neighbors(neighbor, corners_with_index, threshold)
                        if len(neighbors) >= min_neighbors:
                            neighbors.update(extended_neighbors)
                    # if ():
                        cluster.add(neighbor)
                clusters.append(cluster)
    return clusters


#
# def expand_cluster(neighbors: list):
#     for corner in neighbors:
#         if sta


def get_distance(corner1, corner2) -> float:
    distance = math.sqrt((corner1[0] - corner2[0]) ** 2 + (corner1[1] - corner2[1]) ** 2)
    return distance


def mark_blob(image_data, outer_corners: list, rgb_color: list = (255, 0, 0)) -> np.ndarray:
    image_rgb_data = convert_grayscale_to_rgb(image_data)
    x_min = outer_corners[0]
    y_min = outer_corners[1]
    x_max = outer_corners[2]
    y_max = outer_corners[3]
    for x in range(x_min, x_max + 1):
        image_rgb_data[x][y_min] = rgb_color
        image_rgb_data[x][y_max] = rgb_color
    for y in range(y_min, y_max + 1):
        image_rgb_data[x_min][y] = rgb_color
        image_rgb_data[x_max][y] = rgb_color
    return image_rgb_data


def detect_blobs(folder_path: str) -> list:
    result = []
    for filename in os.listdir(folder_path):
        if filename != '.directory':
            filepath = INPUT_DIR + filename
            image = corr.read_grayscale_image_as_array(filepath)
            output_filename = folder_path + 'result_' + filename
            coordinates = harris_corner_detector(image, WINDOW_SIZE, HARRIS_DETECTOR_ALPHA, HARRIS_DETECTOR_THRESHOLD, output_filename, show_corners=True)
            result.append({'file': filename, 'coords': coordinates})
    return result


def main():
    print(detect_blobs(INPUT_DIR))
    print('Done')


if __name__ == "__main__":
    main()
