import scipy.io
from skimage.util import view_as_windows
from skimage.util import pad
import numpy as np

def get_data(file_name):
    return scipy.io.loadmat(file_name)

def get_patches(image, size = 3, use_padding = 'SAME'):

    # add padding for image to make it able to extract the number of patches equal to number of pixels
    # the pad width should eqal (patch size - 1) / 2,  - center pixel will always lay in original image
    nb_padding_pixels = int(size/2 - 0.5)
    # choose symmetric mode
    image_padded = pad(image, nb_padding_pixels, 'symmetric')

    patch_size = (size, size, 103)
    patches = view_as_windows(image_padded, patch_size)
    return patches

def grey_to_lbp(image):

    image_shape = image.shape
    image_width = image_shape[0]
    image_height = image_shape[1]
    image_depth = image_shape[2]

    # print("Shape: {}, Width: {}, Height: {}, Depth: {}".format(image_shape, image_width, image_height, image_depth))

    #get the image shape
    central_pixel = image[int((image_width - 1) / 2), int((image_height - 1) / 2), :]

    # print("Central pixel: {}".format(central_pixel))

    lbp_map = np.zeros(image.shape)
    # dimensions of the image: (image_width,image_height,image_depth)
    # for each of k channels
    for k in range(image_depth):
        # compare each pixel greyscale value with the central pixel
        for i in range(image_width):
            for j in range(image_height):
                if image[i,j,k] >= central_pixel[k]:
                    lbp_map[i,j,k] = 1
                elif image[i,j,k] < central_pixel[k]:
                    lbp_map[i,j,k] = 0
                # setting the central pixel to 0.5 removes the noisy pixel from the middle of a image
                #but LBP map stops to be binary
                #when using it, remove the eqaul sign from the first conditional equation
                # else:
                #     lbp_map[i,j,k] = 0.5

    print('lbp map: {}'.format(lbp_map))

    return lbp_map


def grey_to_clbp(image):

    image_shape = image.shape
    image_width = image_shape[0]
    image_height = image_shape[1]
    image_depth = image_shape[2]

    # print("Shape: {}, Width: {}, Height: {}, Depth: {}".format(image_shape, image_width, image_height, image_depth))

    #get the image shape
    central_pixel = image[int((image_width - 1) / 2), int((image_height - 1) / 2), :]

    # print("Central pixel: {}".format(central_pixel))

    lbp_map = np.zeros(image.shape)
    # dimensions of the image: (image_width,image_height,image_depth)
    # for each of k channels
    for k in range(image_depth):
        # compare each pixel greyscale value with the central pixel
        for i in range(image_width):
            for j in range(image_height):
                if image[i,j,k] >= central_pixel[k]:
                    lbp_map[i,j,k] = 1
                elif image[i,j,k] < central_pixel[k]:
                    lbp_map[i,j,k] = 0
                # setting the central pixel to 0.5 removes the noisy pixel from the middle of a image
                #but LBP map stops to be binary
                #when using it, remove the eqaul sign from the first conditional equation
                # else:
                #     lbp_map[i,j,k] = 0.5

    print('lbp map: {}'.format(lbp_map))

    return lbp_map





def hs_to_grey(image):
    return np.mean(image, axis = 2)
