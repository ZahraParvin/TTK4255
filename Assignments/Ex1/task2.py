import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


## Task 2.1:

image = Image.open( "../data/grass.jpg" )
image.load()
img = np.asarray( image, dtype="int32" )

print("The height is: " + str(img.shape[0]) + " and the width is: " + str(img.shape[1]))

## Task 2.2:
red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]

# Plot the channels
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(red_channel, cmap='jet', interpolation='none')
axes[0].set_title('Red Channel')
axes[0].axis('off')

axes[1].imshow(green_channel, cmap='jet', interpolation='none')
axes[1].set_title('Green Channel')
axes[1].axis('off')

axes[2].imshow(blue_channel, cmap='jet', interpolation='none')
axes[2].set_title('Blue Channel')
axes[2].axis('off')

plt.savefig('task2.2.png')
#plt.show()

## Task 2.3:

threshold = 132  #adjust this based on the image

# Apply thresholding
binary_image = green_channel > threshold

#Display the result
plt.figure(figsize=(6, 6))
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image after Thresholding')
plt.axis('off')
plt.savefig('task2.3.png')
#plt.show()


## Task 2.4:
epsilon = 1e-8
sum_rgb = np.sum(img, axis=2) + epsilon

# Calculate normalized RGB values
r_norm = img[:, :, 0] / sum_rgb
g_norm = img[:, :, 1] / sum_rgb
b_norm = img[:, :, 2] / sum_rgb

# Plot the normalized channels
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(r_norm, cmap='gray')
axes[0].set_title('Normalized Red Channel')
axes[0].axis('off')

axes[1].imshow(g_norm, cmap='gray')
axes[1].set_title('Normalized Green Channel')
axes[1].axis('off')

axes[2].imshow(b_norm, cmap='gray')
axes[2].set_title('Normalized Blue Channel')
axes[2].axis('off')

plt.savefig('task2.4.png')
# plt.show()

## Task 2.5:

threshold = 0.4  # Threshold

binary_image_norm = g_norm > threshold

# Replace above-threshold pixels with magenta in the original image
magenta = [255, 0, 255]  # RGB value for magenta
segmented_image = np.where(binary_image_norm[..., None], magenta, img).astype(np.uint8)

# Plotting the results
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(img.astype(np.uint8))
plt.title('Original Image')
plt.axis('off')

# Binary image
plt.subplot(1, 3, 2)
plt.imshow(binary_image_norm, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

# Segmented image with magenta
plt.subplot(1, 3, 3)
plt.imshow(segmented_image)
plt.title('Segmentation Result')
plt.axis('off')
plt.savefig('task2.5.png')
# plt.show()