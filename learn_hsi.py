import get_data
import numpy as np
import matplotlib.pyplot as plt

mat = get_data.get_data("PaviaU.mat")
# print(mat)

mat_gt = get_data.get_data("PaviaU_gt.mat")
# print(mat_gt

data = mat['paviaU']
ground_truth = mat_gt['paviaU_gt']

print("Shape of the cube: {}".format(data.shape))
print("Shape of the labels: {}".format(ground_truth.shape))

print("Maximum value: {}".format(np.argmax(data)))

print("Index of maximum value: {}".format(np.unravel_index(np.argmax(data), data.shape)))

'''
Crop patches
'''

data_patches_hsi = get_data.get_patches(data, 9)
print(data_patches_hsi[1])
print("Shape of patches array: {}\n".format(data_patches_hsi.shape))

print(data_patches_hsi[0,0,0])

get_data.grey_to_lbp(data_patches_hsi[0,0,0])

# print(get_data._hs_to_grey(data_patches_hsi[8,5,0]))

plt.figure(1)
plt.imshow(get_data.hs_to_grey(data_patches_hsi[90,80,0]), cmap= 'gray')
plt.show()



'''
Layer to LBP
'''


