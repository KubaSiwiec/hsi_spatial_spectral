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

    # print("Shape: {}, Width: {}, Height: {}".format(patch_shape, patch_width, patch_height))

    #get the patch shape
    central_pixel = patch[int((patch_width - 1) / 2), int((patch_height - 1) / 2), :]

    # print("Central pixel: {}".format(central_pixel))




def to_grey(patch):
    return np.mean(patch, axis = 2)
