import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
file_path = 'data/KaggleDB/combretum_07.jpg'
image = cv2.imread(file_path, cv2.IMREAD_COLOR)

# Preprocessing: Increase contrast
# Convert to YUV color space, equalize the histogram of the Y channel, and convert back to BGR
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

# Convert to grayscale for thresholding
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image: keep only the bright areas (presumably the pollen)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Morphological operations to clean up the thresholded image
kernel = np.ones((5, 5), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours of the pollen
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an all black image to draw the contours on
segmented = np.zeros_like(image)

# Draw the contours on the black image
cv2.drawContours(segmented, contours, -1, (0, 255, 0), 2)

# Display the original and segmented images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title('Segmented Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Since the pollen grains are closed contours and we want the biggest one, 
# we will proceed with the following steps:

# Find all contours in the binary image
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming that the largest contour by area is the pollen grain we are interested in
largest_contour = max(contours, key=cv2.contourArea)

# Create an all black mask
mask = np.zeros_like(gray)

# Fill the largest contour with white color in the mask
cv2.drawContours(mask, [largest_contour], -1, 255, -1)

# Bitwise-and mask with the original image to segment the pollen grain
segmented_pollen = cv2.bitwise_and(image, image, mask=mask)

# Show the largest contour mask and the segmented pollen
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap='gray')
plt.title('Largest Contour Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented_pollen, cv2.COLOR_BGR2RGB))
plt.title('Segmented Pollen')
plt.axis('off')

plt.tight_layout()
plt.show()
