import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if not hasattr(np, "sctypes"):
    np.sctypes = {
      "int": [np.int8, np.int16, np.int32, np.int64],
      "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
      "float": [np.float16, np.float32, np.float64],
      "complex": [np.complex64, np.complex128],
      "others": [bool, object, bytes, str, np.void],
    }

from imgaug import augmenters as iaa


def display_augmented_and_original(image_paths, steering_angle):
  random_index = random.randint(0, len(image_paths) - 1)
  image = image_paths[random_index]
  steering_angle = steering_angle[random_index]
  original_image = mpimg.imread(image)
  augmented_image, steering_angle = random_augment(image, steering_angle)
  fig, axs = plt.subplots(1, 2, figsize=(15, 10))
  fig.tight_layout()
  axs[0].imshow(original_image)
  axs[0].set_title("Original Image" + str(steering_angle))
  axs[1].imshow(augmented_image)
  axs[1].set_title("Augmented Image" + str(steering_angle))
  plt.show()


def img_preprocess_no_imread(img):
    img = img[60:135, :, :]
    # The NVIDIA paper recommends YUV rather than RBG
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img  


def zoom(image):
  zoom_func = iaa.Affine(scale=(1.0, 1.3))
  z_image = zoom_func.augment_image(image)
  return z_image


def pan(image_to_pan):
  pan_func = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
  pan_image = pan_func.augment_image(image_to_pan)
  return pan_image


def img_random_brightness(image):
  bright_func = iaa.Multiply((0.2, 1.2))
  bright_image = bright_func.augment_image(image)
  return bright_image


def random_flip(image, steering_angle):
  # 1 - flip horizontally, 0 - vertical, -1 combo of both
  flipped_image = cv2.flip(image, 1)
  steering_angle = - steering_angle
  return flipped_image, steering_angle


def random_augment(image, steering_angle):
  augment_image = mpimg.imread(image)
  if np.random.rand() < 0.5:
    augment_image = zoom(augment_image)
  if np.random.rand() < 0.5:
    augment_image = pan(augment_image)
  if np.random.rand() < 0.5:
    augment_image = img_random_brightness(augment_image)
  if np.random.rand() < 0.5:
    augment_image, steering_angle = random_flip(augment_image, steering_angle)
  return augment_image, steering_angle


def batch_generator(image_paths, steering_angles, batch_size, is_training):
  while True:
    batch_img = []
    batch_steering = []
    for i in range(batch_size):
      random_index = np.random.randint(0, len(image_paths) - 1)
      if is_training:
        img, steering = random_augment(image_paths[random_index], steering_angles[random_index])
      else:
        img = mpimg.imread(image_paths[random_index])
        steering = steering_angles[random_index]
      
      img = img_preprocess_no_imread(img)
      batch_img.append(img)
      batch_steering.append(steering)
      
    yield (np.asarray(batch_img), np.asarray(batch_steering))