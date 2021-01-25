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
    gray = get_grayscale(image)
    thresh = thresholding(gray)
    # cv2.imshow("Result Image", thresh)
    # cv2.waitKey(0)
    return thresh


def print_words(image, config):
    boxes = pytesseract.image_to_data(image, config=config)
    for b in boxes.splitlines()[1:]:
        b = b.split()
        if len(b) == 12:
            if special_match(b[11]):
                print(b[11])


# passing cropped words
if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    img = cv2.imread('words/img52.jpg')
    img = preproccess(img)
    print_words(img,custom_config)

    # gray = get_grayscale(img)
    # thresh = thresholding(gray)
    # opening = opening(gray)
    # canny = canny(gray)
    # img = thresh

    #hImg, wImg = img.shape
    #p_str = pytesseract.image_to_string(img, config=custom_config)

    # for b in boxes.splitlines():
    #     b = b.split(" ")
    #     print(b)
    #     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    #     cv2.rectangle(img, (x,hImg-y),(w,hImg-h),(0,0,255),1)

    # boxes = pytesseract.image_to_data(img, config=custom_config)
    # for b in boxes.splitlines()[1:]:
    #     b = b.split()
    #     if len(b) == 12:
    #         if special_match(b[11]):
    #             print(b[11])

    # cv2.imshow("Result Image", img)
    # cv2.waitKey(0)



