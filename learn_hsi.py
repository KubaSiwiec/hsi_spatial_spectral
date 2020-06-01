from keras_preprocessing.image import ImageDataGenerator
from cv2 import imwrite
import hsi_preprocessing
import learning_models
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import matplotlib.pyplot as plt


mat = hsi_preprocessing.get_data("PaviaU.mat")
# print(mat)

mat_gt = hsi_preprocessing.get_data("PaviaU_gt.mat")
# print(mat_gt

data = mat['paviaU']
ground_truth = mat_gt['paviaU_gt']

print("Shape of the cube: {}".format(data.shape))
print("Shape of the labels_arr: {}".format(ground_truth.shape))
print("Number of classes: {}".format(np.unique(ground_truth)))

print("Maximum value: {}".format(np.argmax(data)))

print("Index of maximum value: {}".format(np.unravel_index(np.argmax(data), data.shape)))


# print(get_data._hs_to_grey(data_patches_hsi[8,5,0]))

# apply lbp and clbp on image
lbp_image = hsi_preprocessing.grey_to_lbp(data)
print('LBP image shape: {}'.format(lbp_image.shape))
# clbp_image = hsi_preprocessing.grey_to_clbp(data)

#crop patches
data_patches_lbp = hsi_preprocessing.get_patches(lbp_image, 5)
print('LBP patches shape: {}'.format(data_patches_lbp.shape))
# data_patches_clbp = hsi_preprocessing.get_patches(clbp_image, 3)

'''
Plotting images
'''

'''
#compare greyscale image with lbp map
plt.figure(1)
plt.imshow(hsi_preprocessing.hs_to_grey(data), cmap='gray')
plt.title("Grayscale image")

plt.figure(2)
plt.imshow(hsi_preprocessing.hs_to_grey(lbp_image), cmap='gray')
plt.title("LBP image")

plt.figure(3)
plt.imshow(hsi_preprocessing.hs_to_grey(clbp_image), cmap='gray')
plt.title("CLBP image")

# present some patches of lbp data
plt.figure(4)
co_x = 0
co_y = 0
plt.imshow(hsi_preprocessing.hs_to_grey(data_patches_lbp[co_x, co_y, 0]), cmap='gray')
plt.title("LBP sample patch of coordinates {}, {}".format(co_x, co_y))

# present some patches of lbp data
plt.figure(5)
co_x = 11
co_y = 107
plt.imshow(hsi_preprocessing.hs_to_grey(data_patches_clbp[co_x, co_y, 0]), cmap='gray')
plt.title("LBP sample patch of coordinates {}, {}".format(co_x, co_y))

plt.figure(6)
plt.imshow(ground_truth)
plt.title('Classes')

plt.show()
'''








'''
Save patches and ground truth into arrays
'''
# patches
patch_lbp_arr = np.asarray(hsi_preprocessing.arr5D_to_list_of_3D_arr(data_patches_lbp))
print('patch_lbp_list len: {}'.format(len(patch_lbp_arr)))

# ground truth
labels_gt = np.asarray(hsi_preprocessing.arr2D_to_list(ground_truth))
print("Labels len: {}".format(len(labels_gt)))
print(np.unique(labels_gt))

'''
ImageDataGenerator and it's instances
'''


val_split = 0.25
X_train, X_val, y_train, y_val = train_test_split(patch_lbp_arr, labels_gt, test_size=val_split, stratify=labels_gt)

patch_lbp_arr = None
labels_gt = None

# patch_lbp_arr = np.concatenate((X_train, X_val))
# labels_gt = np.concatenate((y_train, y_val))

# image_gen = ImageDataGenerator(rotation_range=20,
#                                width_shift_range=0.1,
#                                height_shift_range=0.1,
#                                shear_range=0.1,
#                                zoom_range=0.1,
#                                horizontal_flip=True,
#                                validation_split=0.5,
#                                fill_mode='nearest')
#
# train_generator = image_gen.flow(
#     patch_lbp_arr,
#     labels_gt,
#     shuffle=True,
#     batch_size=256,
#     subset='training')  # set as training data
#
# # data image generator is able to automaticaly split data for training and validation
# validation_generator = image_gen.flow(
#     patch_lbp_arr,
#     labels_gt,
#     shuffle=True,
#     batch_size=256,
#     subset='validation')  # set as validation data


'''
Use image data generator for further preprocessing, split for training and validation
'''
model = learning_models.create_model(None, (5, 5, 103))

# print(model.summary())

model.fit(X_train, y_train, batch_size=256,  epochs=10, validation_data=(X_val, y_val))



'''
Create model
'''



'''
Train and evaluate with apropriate metrics
'''














