import numpy as np
from PIL import Image
from numba import jit, prange

IMAGE_PATH = 'files/file.png'
KERNEL_PATH = 'files/gaussian_kernel.png'
OUTPUT_PATH = 'files/correlation_result.png'
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
        pixels = np.asarray(image).astype('uint8')
        normalized_pixels = normalize_grayscale_pixels(pixels)
        return normalized_pixels
    except IOError:
        print('Input file is not accessible ', input_path)


def save_pixels_as_grayscale_image(pixels, output_path: str):
    denormalized_pixels = denormalize_grayscale_pixels(pixels)
    array = np.asarray(denormalized_pixels).astype(np.uint8)
    image = Image.fromarray(array)
    try:
        image.save(output_path)
        print('File has been successfully saved: ', output_path)
    except IOError:
        print('Cannot save image as a file ', output_path)


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


@jit(nopython=True, parallel=True)
def get_correlation(image_data, kernel_data) -> np.ndarray:
    kernel_data = np.asarray(kernel_data)
    kernel_rows_number = kernel_data.shape[0]
    kernel_columns_number = kernel_data.shape[1]

    image_data = np.asarray(image_data)
    image_rows_number = image_data.shape[0]
    image_columns_number = image_data.shape[1]
    image_row_offset = kernel_rows_number // 2
    image_column_offset = kernel_columns_number // 2

    correlation_shape = (image_rows_number - kernel_rows_number + kernel_rows_number % 2,
                         image_columns_number - kernel_columns_number + kernel_columns_number % 2)
    correlation = [[0.0 for y in range(correlation_shape[1])] for x in range(correlation_shape[0])]

    for image_row in prange(image_row_offset, image_rows_number - image_row_offset):
        for image_column in prange(image_column_offset, image_columns_number - image_column_offset):
            for kernel_row in prange(-image_row_offset, image_row_offset + kernel_rows_number % 2):
                for kernel_column in prange(-image_column_offset, image_column_offset + kernel_columns_number % 2):
                    correlation[image_row - image_row_offset][image_column - image_column_offset] += \
                        kernel_data[kernel_row + image_row_offset][kernel_column + image_column_offset] * \
                        image_data[image_row + kernel_row][image_column + kernel_column]
    return np.array(correlation)


def flip_kernel(kernel_data) -> np.ndarray:
    kernel_data = np.asarray(kernel_data)
    kernel_rows_number = kernel_data.shape[0]
    kernel_columns_number = kernel_data.shape[1]
    kernel_row_max_index = kernel_rows_number - 1
    kernel_column_max_index = kernel_columns_number - 1
    flipped_kernel = [[0.0 for y in range(kernel_columns_number)] for x in range(kernel_rows_number)]
    for row in range(kernel_rows_number):
        for column in range(kernel_columns_number):
            flipped_kernel[row][column] = kernel_data[kernel_row_max_index - row][kernel_column_max_index - column]
    return np.asarray(flipped_kernel)


def get_convolution(image_data, kernel_data) -> np.ndarray:
    flipped_kernel = flip_kernel(kernel_data)
    return get_correlation(image_data, flipped_kernel)


def correlation_to_file(image_path: str, kernel_path: str, output_path: str):
    image = read_grayscale_image_as_array(image_path)
    kernel = read_grayscale_image_as_array(kernel_path)
    image_row_offset = kernel.shape[0] // 2
    image_column_offset = kernel.shape[1] // 2
    padded_image = image_padding(image, image_row_offset, image_column_offset)
    correlation = get_correlation(padded_image, kernel)
    save_pixels_as_grayscale_image(correlation, output_path)


def main():
    correlation_to_file(IMAGE_PATH, KERNEL_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
