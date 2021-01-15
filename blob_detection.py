import os
import math
import numpy as np
import correlation as corr
from PIL import Image, ImageDraw

IMAGE_DIR = 'files/Blobs/'
OUTPUT_DIR = 'files/Blobs/'
IMAGE_PATH = 'files/Blobs/SR11.png'
# IMAGE_PATH = 'files/kernel_old.png'
MARKED_CORNERS_OUTPUT_PATH = 'files/marked_corners_SR11_1.png'
KERNEL_PATH = 'files/kernel.png'
GAUSSIAN_KERNEL_PATH = 'files/Blobs/gaussian_kernel.png'
OUTPUT_PATH = 'files/Blobs/blur_result.png'
LAPLACIAN_DX_DERIVATIVE_PATH = 'files/Blobs/laplacian_dx_derivative.png'
LAPLACIAN_DY_DERIVATIVE_PATH = 'files/Blobs/laplacian_dy_derivative.png'
SOBEL_DX_DERIVATIVE_PATH = 'files/Blobs/sobel_dx_derivative.png'
SOBEL_DY_DERIVATIVE_PATH = 'files/Blobs/sobel_dy_derivative.png'
DX_PLUS_DY_DERIVATIVE_PATH = 'files/Blobs/dx_plus_dy.png'
ADDED_DX_DY_DERIVATIVE_PATH = 'files/Blobs/added_dx_dy.png'
DXxDY_DERIVATIVE_PATH = 'files/Blobs/dx_x_dy.png'
D2X2_DERIVATIVE_PATH = 'files/Blobs/d2x2_derivative.png'
D2Y2_DERIVATIVE_PATH = 'files/Blobs/d2y2_derivative.png'
DXxDX_DERIVATIVE_PATH = 'files/Blobs/dx_x_dx.png'
DYxDY_DERIVATIVE_PATH = 'files/Blobs/dy_x_dy.png'

GAUSSIAN_KERNEL_HEIGHT = 20 #20 #21 for checker board
GAUSSIAN_KERNEL_WIDTH = 20 #20 #21 for checker board
GAUSSIAN_SIGMA = 5 #5 #5 for checker board

WINDOW_SIZE = 3 #8 # 10 in general   or   #3 for checker board
HARRIS_DETECTOR_ALPHA = 0.04 # should be in a range of [0.04; 0.06]
HARRIS_DETECTOR_THRESHOLD = 2000

LAPLACIAN_5POINT_STENCIL = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
LAPLACIAN_9POINT_STENCIL = [[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]
LAPLACIAN_X = [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
LAPLACIAN_Y = np.transpose(LAPLACIAN_X)
SOBEL_X = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
SOBEL_Y = np.transpose(SOBEL_X)


def gaussian_distribution(x: float, y: float, sigma: float) -> float:
    if sigma == 0:
        raise ValueError('Sigma should not be zero otherwise division by zero will occur.')
    power = -((x ** 2) + (y ** 2))/(2 * (sigma ** 2))
    gaussian = math.exp(power) / (2 * math.pi * (sigma ** 2))
    return gaussian


def get_gaussian_grayscale_kernel(kernel_rows_number: int, kernel_columns_number: int, sigma: float):
    if kernel_rows_number <= 0 or kernel_columns_number <= 0:
        raise ValueError('kernel_rows_number and kernel_columns_number parameters should be positive integers.')
    kernel_row_offset = kernel_rows_number // 2
    kernel_column_offset = kernel_columns_number // 2
    kernel_row_range_max = kernel_row_offset + kernel_rows_number % 2  # - 1
    kernel_column_range_max = kernel_column_offset + kernel_columns_number % 2  # - 1
    gaussian_kernel = [[0.0 for y in range(kernel_columns_number)] for x in range(kernel_rows_number)]

    for row in range(-kernel_row_offset, kernel_row_range_max):
        for column in range(-kernel_column_offset, kernel_column_range_max):
            gaussian_kernel[row + kernel_row_offset][column + kernel_column_offset] = gaussian_distribution(row, column, sigma)

    return np.asarray(gaussian_kernel)


def save_gaussian_kernel_as_image(kernel, output_path):
    kernel = np.asarray(kernel)
    kernel_rows_number = kernel.shape[0]
    kernel_columns_number = kernel.shape[1]
    kernel_center_row_index = kernel_rows_number // 2
    kernel_center_column_index = kernel_columns_number // 2
    grayscale_gaussian_kernel = (kernel / kernel[kernel_center_row_index][kernel_center_column_index]) * (corr.GRAYSCALE_COLORS_NUMBER - 1)
    # corr.save_array_as_grayscale_image(grayscale_gaussian_kernel, output_path)


def blur_image(image, kernel_rows_number: int, kernel_columns_number: int, sigma: float):
    kernel_row_offset = kernel_rows_number // 2
    kernel_column_offset = kernel_columns_number // 2
    padded_image = corr.image_padding(image, kernel_row_offset, kernel_column_offset)
    gaussian_kernel = get_gaussian_grayscale_kernel(kernel_rows_number, kernel_columns_number, sigma)
    # save_gaussian_kernel_as_image(gaussian_kernel, GAUSSIAN_KERNEL_PATH)
    return corr.get_convolution(padded_image, gaussian_kernel)


def get_laplacian_dx_plus_dy(image):
    kernel_offset = len(LAPLACIAN_5POINT_STENCIL) // 2
    padded_image = corr.image_padding(image, kernel_offset, kernel_offset)
    image_derivative = corr.get_convolution(padded_image, LAPLACIAN_5POINT_STENCIL)
    return image_derivative


def get_laplacian_dx(image):
    kernel_offset = len(LAPLACIAN_X) // 2
    padded_image = corr.image_padding(image, kernel_offset, kernel_offset)
    image_derivative = corr.get_convolution(padded_image, LAPLACIAN_X)
    return image_derivative


def get_laplacian_dy(image):
    kernel_offset = len(LAPLACIAN_Y) // 2
    padded_image = corr.image_padding(image, kernel_offset, kernel_offset)
    image_derivative = corr.get_convolution(padded_image, LAPLACIAN_Y)
    return image_derivative


def get_sobel_dx(image):
    kernel_offset = len(SOBEL_X) // 2
    padded_image = corr.image_padding(image, kernel_offset, kernel_offset)
    image_derivative = corr.get_convolution(padded_image, SOBEL_X)
    return image_derivative


def get_sobel_dy(image):
    kernel_offset = len(SOBEL_Y) // 2
    padded_image = corr.image_padding(image, kernel_offset, kernel_offset)
    image_derivative = corr.get_convolution(padded_image, SOBEL_Y)
    return image_derivative


def get_harris_cornerness(image, window_size=WINDOW_SIZE, alpha=HARRIS_DETECTOR_ALPHA):
    image_rows_number = image.shape[0]
    image_columns_number = image.shape[1]
    window_size_offset = window_size // 2
    window_size_max_index = window_size // 2 + window_size % 2

    # padded_image = corr.image_padding(image, image_row_offset=GAUSSIAN_KERNEL_HEIGHT, image_column_offset=GAUSSIAN_KERNEL_WIDTH)
    # blurred_image = blur_image(padded_image, GAUSSIAN_KERNEL_HEIGHT, GAUSSIAN_KERNEL_WIDTH, GAUSSIAN_SIGMA)


    # blurred_image = blur_image(image, GAUSSIAN_KERNEL_HEIGHT, GAUSSIAN_KERNEL_WIDTH, GAUSSIAN_SIGMA)
    blurred_image = image # skip blurring

    # corr.save_array_as_grayscale_image(blurred_image, OUTPUT_PATH)

    # image_dx = get_sobel_dx(blurred_image)
    image_dx = get_laplacian_dx(blurred_image)
    # corr.save_array_as_grayscale_image(image_dx, SOBEL_DX_DERIVATIVE_PATH)

    # image_dy = get_sobel_dy(blurred_image)
    image_dy = get_laplacian_dy(blurred_image)
    # corr.save_array_as_grayscale_image(image_dy, SOBEL_DY_DERIVATIVE_PATH)

    dx_squared = np.multiply(image_dx, image_dx)
    dy_squared = np.multiply(image_dy, image_dy)
    dx_times_dy = np.multiply(image_dx, image_dy)

    # padded_dx_squared = corr.image_padding(dx_squared, window_size, window_size)
    # padded_dy_squared = corr.image_padding(dy_squared, window_size, window_size)
    # padded_dx_times_dy = corr.image_padding(dx_times_dy, window_size, window_size)

    cornerness = [[0.0 for y in range(image_columns_number)] for x in range(image_rows_number)]

    # TODO: Try to convolve image with gaussian 9x9 kernel instead of calculating sum in a window. Resut will be rotation invariant

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
                    # dx_squared_sum += padded_dx_squared[row + window_row][column + window_column]
                    # dy_squared_sum += padded_dy_squared[row + window_row][column + window_column]
                    # dx_times_dy_sum += padded_dx_times_dy[row + window_row][column + window_column]
            det = dx_squared_sum * dy_squared_sum - (dx_times_dy_sum ** 2)
            trace = dx_squared_sum + dy_squared_sum
            cornerness[row][column] = det - alpha * (trace ** 2)

    return cornerness #


def non_maximum_suppression(cornerness, threshold, window_radius):
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
                    column2 = column -1
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
                            corners.append((row, column, cornerness[row][column]))
                column = column1

    return corners

def convert_grayscale_to_rgb(image_data):
    image_data = np.asarray(image_data)
    if len(image_data.shape) == 3:
        return image_data
    if len(image_data.shape) == 2:
        image_rows_number = image_data.shape[0]
        image_columns_number = image_data.shape[1]
        image_rgb_data = np.zeros((image_rows_number, image_columns_number, 3), 'uint8')
        # image_rgb_data = [[[0.0 for z in range(3)] for y in range(image_columns_number)] for x in range(image_rows_number)]
        for row in range(image_rows_number):
            for column in range(image_columns_number):
                image_rgb_data[row][column] = [image_data[row][column], image_data[row][column], image_data[row][column]]
                # image_rgb_data[row][column][0] = image_data
                # image_rgb_data[row][column][1] = image_data
                # image_rgb_data[row][column][2] = image_data
    return np.asarray(image_rgb_data).astype(np.uint8)


def mark_corners(image, corners):
    mark_radius = 1
    image_rgb_data = convert_grayscale_to_rgb(image)
    image_rows_number = image_rgb_data.shape[0]
    image_columns_number = image_rgb_data.shape[1]
    for corner in corners:
        corner_row = corner[0]
        corner_column = corner[1]
        # corner_row = corner[0] - GAUSSIAN_KERNEL_HEIGHT // 2 # TODO: Just guessing - uncomment if blur is used
        # corner_column = corner[1] - GAUSSIAN_KERNEL_WIDTH // 2 # TODO: Just guessing - uncomment if blur is used
        for mark_row in range(-mark_radius, mark_radius):
            for mark_column in range(-mark_radius, mark_radius):
                if 0 <= corner_row + mark_row < image_rows_number and 0 <= corner_column + mark_column < image_columns_number:
                    image_rgb_data[corner_row + mark_row][corner_column + mark_column] = [0, 255, 0] #RED
    # draw = ImageDraw.Draw(image)
    # for corner in corners:
        # bounding_box = (corner[0] - mark_radius, corner[1] - mark_radius, corner[0] + mark_radius, corner[1 + mark_radius])
        # draw = draw.ellipse(bounding_box, fill=128)
    return image_rgb_data


def harris_corner_detector(image, window_size, alpha, threshold, output_path):
    cornerness = get_harris_cornerness(image, window_size, alpha)
    corners = non_maximum_suppression(cornerness, threshold, window_size * 2) # or don't * 2
    outer_corners = get_outer_corners(corners)
    image_rgb_data = mark_blob(image, outer_corners)
    image_rgb_data = mark_corners(image_rgb_data, corners)
    image = Image.fromarray(image_rgb_data)
    try:
        image.save(output_path)
    except IOError:
        print('Cannot save image as a file ', output_path)
        exit


def get_outer_corners(corners):
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
    return [(x_min, y_min), (x_max, y_max)]


def mark_blob(image, outer_corners):
    image_rgb_data = convert_grayscale_to_rgb(image)
    x_min = outer_corners[0][0]
    y_min = outer_corners[0][1]
    x_max = outer_corners[1][0]
    y_max = outer_corners[1][1]
    for x in range(x_min, x_max + 1):
        image_rgb_data[x][y_min] = [255, 0, 0]
        image_rgb_data[x][y_max] = [255, 0, 0]
    for y in range(y_min, y_max + 1):
        image_rgb_data[x_min][y] = [255, 0, 0]
        image_rgb_data[x_max][y] = [255, 0, 0]
    return image_rgb_data


def main():
    for filename in os.listdir(IMAGE_DIR):
        if filename != '.directory':
            filepath = IMAGE_DIR + filename
            image = corr.read_grayscale_image_as_array(filepath)
            output_filename = OUTPUT_DIR + 'corners_' + filename
            harris_corner_detector(image, WINDOW_SIZE, HARRIS_DETECTOR_ALPHA, HARRIS_DETECTOR_THRESHOLD, output_filename)
    print('Done')
    # blurred_image = blur_image(image, GAUSSIAN_KERNEL_HEIGHT, GAUSSIAN_KERNEL_WIDTH, GAUSSIAN_SIGMA)
    # corr.save_array_as_grayscale_image(blurred_image, OUTPUT_PATH)
    #
    # image_dx_plus_dy = get_laplacian_dx_plus_dy(blurred_image)
    # corr.save_array_as_grayscale_image(image_dx_plus_dy, DX_PLUS_DY_DERIVATIVE_PATH)
    #
    # image_laplacian_dx = get_laplacian_dx(blurred_image)
    # corr.save_array_as_grayscale_image(image_laplacian_dx, LAPLACIAN_DX_DERIVATIVE_PATH)
    #
    # image_sobel_dx = get_sobel_dx(blurred_image)
    # corr.save_array_as_grayscale_image(image_sobel_dx, SOBEL_DX_DERIVATIVE_PATH)
    #
    # image_sobel_dy = get_sobel_dy(blurred_image)
    # corr.save_array_as_grayscale_image(image_sobel_dy, SOBEL_DY_DERIVATIVE_PATH)
    #
    # image_laplacian_dy = get_laplacian_dy(blurred_image)
    # corr.save_array_as_grayscale_image(image_laplacian_dy, LAPLACIAN_DY_DERIVATIVE_PATH)
    #
    # multiplied_dx_dy = np.multiply(image_laplacian_dx, image_laplacian_dy)
    # corr.save_array_as_grayscale_image(multiplied_dx_dy, DXxDY_DERIVATIVE_PATH)
    #
    # multiplied_dx_dx = np.multiply(image_laplacian_dx, image_laplacian_dx)
    # corr.save_array_as_grayscale_image(multiplied_dx_dx, DXxDX_DERIVATIVE_PATH)
    #
    # multiplied_dy_dy = np.multiply(image_laplacian_dy, image_laplacian_dy)
    # corr.save_array_as_grayscale_image(multiplied_dy_dy, DYxDY_DERIVATIVE_PATH)
    #
    # added_dx_dy = np.add(image_laplacian_dx, image_laplacian_dy)
    # corr.save_array_as_grayscale_image(added_dx_dy, ADDED_DX_DY_DERIVATIVE_PATH)
    #
    # image_d2x2_derivative = get_laplacian_dx(image_laplacian_dx)
    # corr.save_array_as_grayscale_image(image_d2x2_derivative, D2X2_DERIVATIVE_PATH)
    #
    # image_d2y2_derivative = get_laplacian_dy(image_laplacian_dy)
    # corr.save_array_as_grayscale_image(image_d2y2_derivative, D2Y2_DERIVATIVE_PATH)


if __name__ == "__main__":
    main()