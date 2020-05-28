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

grey_scale_data = np.zeros([610, 340])


for i in range(grey_scale_data.shape[0]):
    for j in range(grey_scale_data.shape[1]):
        sum_pixel = np.sum(data[i, j, :])
        # print(sum_pixel)
        grey_scale_data[i, j] = int(sum_pixel / 103)
print(grey_scale_data)

print(np.argmax(grey_scale_data))

'''
Crop patches
'''

data_patches_hsi = get_data.get_patches(data, 3)
print(data_patches_hsi[1])
print("Shape of patches array: {}\n".format(data_patches_hsi.shape))

print(data_patches_hsi[0,0,0])



'''
Layer to LBP
'''





plt.figure(1)
plt.imshow(grey_scale_data, cmap="gray")    #may scale the color to maximum 255 or 2048
plt.show()


channels = []
for channel_index in range(data.shape[2]):
    # print(data[:, :, channel_index])

    channels.append(data[:, :, channel_index])

# print(channels[0])

#print(data[:, :, 58])

