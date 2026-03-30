import cv2
import numpy as np
import sys

# Read the image
image_path = 'aces_relations.png'
img = cv2.imread(image_path)

if img is None:
    print(f"Failed to load {image_path}")
    sys.exit(1)

# Convert to HSV color space for better color segmentation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for green color in HSV
# Hue for green is around 60. We take from 35 to 85
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create a mask for green pixels
mask = cv2.inRange(hsv, lower_green, upper_green)

# Replace green pixels with white (255, 255, 255)
img[mask > 0] = [255, 0, 0]

# Save the modified image back
cv2.imwrite(image_path, img)
print("Successfully replaced green pixels with white.")
