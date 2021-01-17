import numpy as np
import image as im
from numba import jit, prange


IMAGE_PATH = 'files/file.png'
KERNEL_PATH = 'files/gaussian_kernel.png'
OUTPUT_PATH = 'files/correlation_result.png'


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
    image = im.read_grayscale_image_as_array(image_path)
    kernel = im.read_grayscale_image_as_array(kernel_path)
    image_row_offset = kernel.shape[0] // 2
    image_column_offset = kernel.shape[1] // 2
    padded_image = im.image_padding(image, image_row_offset, image_column_offset)
    correlation = get_correlation(padded_image, kernel)
    im.save_pixels_as_grayscale_image(correlation, output_path)


def main():
    correlation_to_file(IMAGE_PATH, KERNEL_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
