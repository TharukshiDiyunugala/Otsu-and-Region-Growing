import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Create a synthetic image with 3 pixel values ---
image = np.zeros((100, 100), dtype=np.uint8)
image[20:50, 20:50] = 128  # Object 1
image[60:90, 60:90] = 255  # Object 2

# --- Add Gaussian noise ---
mean = 0
stddev = 20
noise = np.random.normal(mean, stddev, image.shape).astype(np.int16)
noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# --- Optional: Apply Gaussian smoothing ---
smoothed = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# --- Apply Otsu’s thresholding ---
_, otsu_result = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Show the results ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Noisy")
plt.imshow(noisy_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Otsu Result")
plt.imshow(otsu_result, cmap='gray')

plt.tight_layout()
plt.show()
