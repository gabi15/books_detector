from skimage import io
from skimage.color import rgb2gray
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

#original_img = io.imread(r"D:\Gabrysia\studia\books_detection\books.jpg")
#original_img = io.imread(r"images\books.jpg")
# grayscale_img = rgb2gray(original_img)
#
# canny_img = feature.canny(grayscale_img)
#
# # Classic straight-line Hough transform
# # Set a precision of 0.5 degree.
# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
# h, theta, d = hough_line(canny_img)
#
#
# # Generating figure 1
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# ax = axes.ravel()
#
# ax[0].imshow(canny_img, cmap="gray")
# ax[0].set_title('Input image')
# ax[0].set_axis_off()
#
# ax[1].imshow(canny_img, cmap="gray")
# origin = np.array((0, canny_img.shape[1]))
# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
#     ax[1].plot(origin, (y0, y1), '-r')
# ax[1].set_xlim(origin)
# ax[1].set_ylim((canny_img.shape[0], 0))
# ax[1].set_axis_off()
# ax[1].set_title('Detected lines')
#
# plt.tight_layout()
# plt.show()

# Read image
img = cv2.imread(r"images\books.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (1000, 670), interpolation=cv2.INTER_AREA)

# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)

# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, minLineLength=300, maxLineGap=200)

# Sort lines by x values
new_list = sorted(lines, key=lambda line: line[0][0])


def count_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = abs(math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180.0 / math.pi)
    return angle


# save only lines close to vertical
new_list = [line for line in new_list if 70 <= count_angle(line) <= 90]

# find indexes of new_list that will be excluded
exclude = []
for i in range(len(new_list)-1):
    if abs(new_list[i][0][0]-new_list[i+1][0][0]) < 35:
        exclude.append(i+1)

# exclude multiple lines that should be represented by a single line
new_list = [i for j, i in enumerate(new_list) if j not in exclude]

cropped_images = []

# cutting single books from a shelf
for i in range(len(new_list)-1):
    x1 = new_list[i][0][0]
    x2 = new_list[i][0][2]
    x3 = new_list[i+1][0][0]
    x4 = new_list[i+1][0][2]
    crop_img = img[:, min(x1, x2): max(x3, x4)]
    cropped_images.append(crop_img)

# saving images
for i, image in enumerate(cropped_images):
    filename = "after_cropping/img"+str(i)+".jpg"
    print(filename)
    cv2.imwrite(filename, image)

# putting lines on original image to see how algorithm worked
for i, line in enumerate(new_list):
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

# Show result
cv2.imshow("Result Image", img)
cv2.waitKey(0)


