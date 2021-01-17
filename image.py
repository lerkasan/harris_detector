import sys
import numpy as np
from PIL import Image


MAX_GRAYSCALE_COLOR_VALUE = 255
GRAYSCALE_COLORS_NUMBER = MAX_GRAYSCALE_COLOR_VALUE + 1


def normalize_grayscale_pixels(pixels) -> np.ndarray:
    data = np.asarray(pixels).astype(float)
    rows_number = data.shape[0]
    columns_number = data.shape[1]
    for row in range(rows_number):
        for column in range(columns_number):
            data[row][column] = data[row][column] / MAX_GRAYSCALE_COLOR_VALUE
    return data


def denormalize_grayscale_pixels(pixels) -> np.ndarray:
    data = np.asarray(pixels).astype(float)
    rows_number = data.shape[0]
    columns_number = data.shape[1]
    for row in range(rows_number):
        for column in range(columns_number):
            data[row][column] = (data[row][column] * MAX_GRAYSCALE_COLOR_VALUE) % GRAYSCALE_COLORS_NUMBER
    return data


def read_grayscale_image_as_array(input_path: str) -> np.ndarray:
    #Open the image in grayscale mode
    try:
        image = Image.open(input_path).convert('L')
        if not image:
            raise IOError('Input file is empty')
        pixels = np.asarray(image).astype('uint8')
        normalized_pixels = normalize_grayscale_pixels(pixels)
        return normalized_pixels
    except IOError:
        print('Input file is empty or not accessible ', input_path)
        sys.exit(1)


def save_pixels_as_grayscale_image(pixels, output_path: str):
    denormalized_pixels = denormalize_grayscale_pixels(pixels)
    array = np.asarray(denormalized_pixels).astype(np.uint8)
    image = Image.fromarray(array)
    try:
        image.save(output_path)
        print('File has been successfully saved: ', output_path)
    except IOError:
        print('Cannot save image as a file ', output_path)


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


def image_padding(image_data, image_row_offset: int, image_column_offset: int) -> np.ndarray:
    if image_row_offset < 0 or image_column_offset < 0:
        raise ValueError('image_row_offset and image_column_offset parameters should be non-negative integers.')
    image_data = np.asarray(image_data)
    image_rows_number = image_data.shape[0]
    image_columns_number = image_data.shape[1]
    padded_image_shape = (image_rows_number + image_row_offset * 2, image_columns_number + image_column_offset * 2)
    padded_image = [[1.0 for y in range(padded_image_shape[1])] for x in range(padded_image_shape[0])]
    for row in range(image_rows_number):
        for column in range(image_columns_number):
            padded_image[row + image_row_offset][column + image_column_offset] = image_data[row][column]
    return np.asarray(padded_image)
