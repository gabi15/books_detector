import pytesseract
import cv2
import numpy as np
import re

# https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/ useful link

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def special_match(strg, search=re.compile(r"[^a-zA-Z0-9.']").search):
    return not bool(search(strg))


def preproccess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1))
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.medianBlur(img, 5)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    sum = img[0][0]/255 + img[0][-1]/255 + img[-1][0]/255 + img[-1][-1]/255
    if sum < 2 :
        img = np.invert(img)
    # cv2.imshow("Result Image", img)
    # cv2.waitKey(0)
    return img


def print_words(image, config):
    boxes = pytesseract.image_to_data(image, config=config)
    result = ''
    for b in boxes.splitlines()[1:]:
        b = b.split()
        if len(b) == 12 and len(b[11])>2:
            result += b[11] + ' '
    return result


