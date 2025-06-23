import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Create the same synthetic image ---
image = np.zeros((100, 100), dtype=np.uint8)
image[20:50, 20:50] = 128  # Object 1
image[60:90, 60:90] = 255  # Object 2

# --- Region Growing Algorithm ---
def region_growing(img, seed_point, threshold=15):
    height, width = img.shape
    segmented = np.zeros_like(img)
    visited = np.zeros_like(img, dtype=bool)

    seed_x, seed_y = seed_point
    seed_value = int(img[seed_x, seed_y])

    stack = [(seed_x, seed_y)]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue

        visited[x, y] = True
        pixel_value = int(img[x, y])

        if abs(pixel_value - seed_value) <= threshold:
            segmented[x, y] = 255
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                        stack.append((nx, ny))

    return segmented

# --- Define seed point and run segmentation ---
seed = (25, 25)  # inside object 1
result = region_growing(image, seed, threshold=20)

# --- Show result ---
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Region Grown")
plt.imshow(result, cmap='gray')

plt.tight_layout()
plt.show()
