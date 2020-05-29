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

def grey_to_lbp(patch):

    patch_shape = patch.shape
    patch_width = patch_shape[0]
    patch_height = patch_shape[1]
    patch_depth = patch_shape[2]

    # print("Shape: {}, Width: {}, Height: {}, Depth: {}".format(patch_shape, patch_width, patch_height, patch_depth))

    #get the patch shape
    central_pixel = patch[int((patch_width - 1) / 2), int((patch_height - 1) / 2), :]

    # print("Central pixel: {}".format(central_pixel))

    lbp_map = np.zeros(patch.shape)
    # dimensions of the patch: (patch_width,patch_height,patch_depth)
    # for each of k channels
    for k in range(patch_depth):
        # compare each pixel greyscale value with the central pixel
        for i in range(patch_width):
            for j in range(patch_height):
                if patch[i,j,k] >= central_pixel[k]:
                    lbp_map[i,j,k] = 1
                elif patch[i,j,k] < central_pixel[k]:
                    lbp_map[i,j,k] = 0
                # setting the central pixel to 0.5 removes the noisy pixel from the middle of a patch
                #but LBP map stops to be binary
                #when using it, remove the eqaul sign from the first conditional equation
                # else:
                #     lbp_map[i,j,k] = 0.5

    print('lbp map: {}'.format(lbp_map))

    return lbp_map


def grey_to_clbp(patch):

    patch_shape = patch.shape
    patch_width = patch_shape[0]
    patch_height = patch_shape[1]
    patch_depth = patch_shape[2]

    # print("Shape: {}, Width: {}, Height: {}, Depth: {}".format(patch_shape, patch_width, patch_height, patch_depth))

    #get the patch shape
    central_pixel = patch[int((patch_width - 1) / 2), int((patch_height - 1) / 2), :]

    # print("Central pixel: {}".format(central_pixel))

    lbp_map = np.zeros(patch.shape)
    # dimensions of the patch: (patch_width,patch_height,patch_depth)
    # for each of k channels
    for k in range(patch_depth):
        # compare each pixel greyscale value with the central pixel
        for i in range(patch_width):
            for j in range(patch_height):
                if patch[i,j,k] >= central_pixel[k]:
                    lbp_map[i,j,k] = 1
                elif patch[i,j,k] < central_pixel[k]:
                    lbp_map[i,j,k] = 0
                # setting the central pixel to 0.5 removes the noisy pixel from the middle of a patch
                #but LBP map stops to be binary
                #when using it, remove the eqaul sign from the first conditional equation
                # else:
                #     lbp_map[i,j,k] = 0.5

    print('lbp map: {}'.format(lbp_map))

    return lbp_map





def hs_to_grey(patch):
    return np.mean(patch, axis = 2)
