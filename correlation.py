import numpy as np
from PIL import Image
from numba import jit, prange

IMAGE_PATH = 'files/file5.png'
KERNEL_PATH = 'files/kernel5.png'
OUTPUT_PATH = 'files/correlation_result5b.png'
GRAYSCALE_COLORS_NUMBER = 256


def read_grayscale_image_as_array(input_path: str):
    #Open the image in grayscale mode
    try:
        image = Image.open(input_path).convert('L')
    except IOError:
        print('Input file is not accessible ', input_path)
        exit
    data = np.asarray(image).astype(float)
    # print(type(data))
    # print(data.shape)
    return data


def save_array_as_grayscale_image(
        # inputList=None,
        # inputArray=None,
        array, output_path: str):
    # if not array:
    #     raise ValueError('An array is empty: ', array)
    np_array = np.asarray(array).astype(np.uint8)
    image2 = Image.fromarray(np_array)
    try:
        image2.save(output_path)
    except IOError:
        print('Cannot save image as a file ', output_path)
        exit
    # print(type(image2))
    print('File has been successfully saved: ', output_path)


def image_padding(image_data, image_row_offset: int, image_column_offset: int):
    if image_row_offset < 0 or image_column_offset < 0:
        raise ValueError('image_row_offset and image_column_offset parameters should be non-negative integers.')

    image_rows_number = image_data.shape[0]
    image_columns_number = image_data.shape[1]
    padded_image_shape = (image_rows_number + image_row_offset * 2, image_columns_number + image_column_offset * 2)

    padded_image = [[0.0 for y in range(padded_image_shape[1])] for x in range(padded_image_shape[0])]
    # padded_image = [[255.0 for y in range(padded_image_shape[1])] for x in range(padded_image_shape[0])]

    for row in range(image_rows_number):
        for column in range(image_columns_number):
            padded_image[row + image_row_offset][column + image_column_offset] = image_data[row][column]

    return np.asarray(padded_image)


def add_grayscale_pixels(pixel1_value: float, pixel2_value: float) -> float:
    if pixel1_value < 0 or pixel1_value > 255 or pixel2_value < 0 or pixel2_value > 255:
        raise ValueError('pixel_value parameters should be in a range of [0; 255]')
    return (pixel1_value + pixel2_value) % GRAYSCALE_COLORS_NUMBER


def multiply_grayscale_pixels(pixel1_value: float, pixel2_value: float) -> float:
    if pixel1_value < 0 or pixel1_value > 255 or pixel2_value < 0 or pixel2_value > 255:
        raise ValueError('pixel_value parameters should be in a range of [0; 255]')
    return (pixel1_value * pixel2_value) % GRAYSCALE_COLORS_NUMBER


@jit(nopython=True, parallel=True)
def get_correlation(image, kernel):

    #TODO: check that image and kernel np.arrays are not empty

    kernel_rows_number = kernel.shape[0]
    kernel_columns_number = kernel.shape[1]

    image_rows_number = image.shape[0]
    image_columns_number = image.shape[1]
    image_row_offset = kernel_rows_number // 2
    image_column_offset = kernel_columns_number // 2

    correlation_shape = (image_rows_number - kernel_rows_number + kernel_rows_number % 2,
                         image_columns_number - kernel_columns_number + kernel_columns_number % 2)

    correlation = [[0.0 for y in range(correlation_shape[1])] for x in range(correlation_shape[0])]

    for image_row in prange(image_row_offset, image_rows_number - image_row_offset):
        # print("Image row: ", image_row)
        for image_column in prange(image_column_offset, image_columns_number - image_column_offset):
            for kernel_row in prange(-image_row_offset, image_row_offset + kernel_rows_number % 2):
                for kernel_column in prange(-image_column_offset, image_column_offset + kernel_columns_number % 2):
                    # print(f"correlation[{image_row - image_row_offset}][{image_column - image_column_offset}] += kernel[{kernel_row + image_row_offset}][{kernel_column + image_column_offset}] * image[{image_row + kernel_row}][{image_column + kernel_column}]")
                    # multiplied_pixels = multiply_grayscale_pixels(kernel[kernel_row + image_row_offset][kernel_column + image_column_offset], image[image_row + kernel_row][image_column + kernel_column])
                    # correlation[image_row - image_row_offset][image_column - image_column_offset] = add_grayscale_pixels(correlation[image_row - image_row_offset][image_column - image_column_offset], multiplied_pixels)
                    correlation[image_row - image_row_offset][image_column - image_column_offset] = \
                        (correlation[image_row - image_row_offset][image_column - image_column_offset] +
                         kernel[kernel_row + image_row_offset][kernel_column + image_column_offset] *
                         image[image_row + kernel_row][image_column + kernel_column]) % GRAYSCALE_COLORS_NUMBER
    # print('Correlation has been successfully calculated ')
    return np.array(correlation)


def flip_kernel(kernel):

    # TODO: check that kernel np.array is not empty
    kernel = np.asarray(kernel)
    kernel_rows_number = kernel.shape[0]
    kernel_columns_number = kernel.shape[1]
    kernel_row_max_index = kernel_rows_number - 1
    kernel_column_max_index = kernel_columns_number - 1
    flipped_kernel = [[0.0 for y in range(kernel_columns_number)] for x in range(kernel_rows_number)]
    for row in range(kernel_rows_number):
        for column in range(kernel_columns_number):
            flipped_kernel[row][column] = kernel[kernel_row_max_index - row][kernel_column_max_index - column]
    return np.asarray(flipped_kernel)


def get_convolution(image, kernel):

    # TODO: check that image and kernel np.arrays are not empty

    flipped_kernel = flip_kernel(kernel)
    return get_correlation(image, flipped_kernel)


def main():
    image = read_grayscale_image_as_array(IMAGE_PATH)
    kernel = read_grayscale_image_as_array(KERNEL_PATH)
    image_row_offset = kernel.shape[0] // 2
    image_column_offset = kernel.shape[1] // 2
    padded_image = image_padding(image, image_row_offset, image_column_offset)
    correlation = get_correlation(padded_image, kernel)
    save_array_as_grayscale_image(correlation, OUTPUT_PATH)


if __name__ == "__main__":
    main()
