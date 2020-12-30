import numpy as np
import cv2
import os

def load_images_from_folder(folder_path, is_masked):# e.g. "image/mask/correct"
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            resized_img = cv2.resize(img, (224, 224))
            images.append(resized_img)
    n_img = len(images)
    if is_masked:
        labels = np.ones((n_img, 1))
    else:
        labels = np.zeros((n_img, 1))
    images = np.array(images)
    return images, labels

# Load images and downscale them to 224 * 224
mask_correct_img, mask_correct_labels = load_images_from_folder("image/mask/correct", True)
mask_incorrect_img, mask_incorrect_labels = load_images_from_folder("image/mask/incorrect", True)
no_mask_images,  no_mask_labels = load_images_from_folder("image/no_mask", False)
mask_images = np.concatenate((mask_correct_img, mask_incorrect_img), axis=0)
mask_labels = np.concatenate((mask_correct_labels, mask_incorrect_labels), axis=0)

all_images = np.concatenate((mask_images, no_mask_images), axis=0)
all_labels = np.concatenate((mask_labels, no_mask_labels), axis=0)
print(all_images.shape)
print(all_labels.shape)
np.save("images.npy", all_images)
np.save("labels.npy", all_labels)
