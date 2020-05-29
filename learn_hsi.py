import hsi_preprocessing
import numpy as np
import matplotlib.pyplot as plt

mat = hsi_preprocessing.get_data("PaviaU.mat")
# print(mat)

mat_gt = hsi_preprocessing.get_data("PaviaU_gt.mat")
# print(mat_gt

data = mat['paviaU']
ground_truth = mat_gt['paviaU_gt']

print("Shape of the cube: {}".format(data.shape))
print("Shape of the labels: {}".format(ground_truth.shape))

print("Maximum value: {}".format(np.argmax(data)))

print("Index of maximum value: {}".format(np.unravel_index(np.argmax(data), data.shape)))


# print(get_data._hs_to_grey(data_patches_hsi[8,5,0]))

# apply lbp on image
lbp_image = hsi_preprocessing.grey_to_lbp(data)

#crop patches
data_patches_lbp = hsi_preprocessing.get_patches(data, 3)

#compare greyscale image with lbp map
plt.figure(1)
plt.imshow(hsi_preprocessing.hs_to_grey(data), cmap='gray')
plt.title("Grayscale image")

plt.figure(2)
plt.imshow(hsi_preprocessing.hs_to_grey(lbp_image), cmap='gray')
plt.title("LBP image")

# present some patches of lbp data
plt.figure(3)
co_x = 0
co_y = 0
plt.imshow(hsi_preprocessing.hs_to_grey(data_patches_lbp[co_x, co_y, 0]), cmap='gray')
plt.title("LBP sample patch of coordinates {}, {}".format(co_x, co_y))












plt.show()







