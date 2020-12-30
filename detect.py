from skimage import io
from skimage.color import rgb2gray
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
import numpy as np

#original_img = io.imread(r"D:\Gabrysia\studia\books_detection\books.jpg")
original_img = io.imread(r"images\books.jpg")
grayscale_img = rgb2gray(original_img)

canny_img = feature.canny(grayscale_img)

# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
h, theta, d = hough_line(canny_img)


# Generating figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(canny_img, cmap="gray")
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(canny_img, cmap="gray")
origin = np.array((0, canny_img.shape[1]))
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[1].plot(origin, (y0, y1), '-r')
ax[1].set_xlim(origin)
ax[1].set_ylim((canny_img.shape[0], 0))
ax[1].set_axis_off()
ax[1].set_title('Detected lines')

plt.tight_layout()
plt.show()


