# import cv2
# import numpy as np
# import pytesseract
#
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# img = cv2.imread('after_cropping/img0.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower = np.array([0, 0, 218])
# upper = np.array([157, 54, 255])
# mask = cv2.inRange(hsv, lower, upper)
#
# # Create horizontal kernel and dilate to connect text characters
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
# dilate = cv2.dilate(mask, kernel, iterations=5)
#
# # Find contours and filter using aspect ratio
# # Remove non-text contours by filling in the contour
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     ar = w / float(h)
#     if ar < 5:
#         cv2.drawContours(dilate, [c], -1, (0,0,0), -1)
#
#
# # Bitwise dilated image with mask, invert, then OCR
# result = 255 - cv2.bitwise_and(dilate, mask)
# data = pytesseract.image_to_string(result, lang='eng',config='--psm 6')
# print(data)
# cv2.imshow("Result Image", dilate)
# cv2.waitKey(0)
import cv2
import numpy as np

large1 = cv2.imread('images/img0.jpg')
small1 = cv2.pyrDown(large1)
large2 = cv2.imread('after_cropping/img0.jpg')
small2 = cv2.pyrDown(large2)
#cv2.imshow("downsized", rgb)



def text_area(rgb):
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #For opencv 3+ comment the previous line and uncomment the following line
    #_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

    cv2.imshow('rects', rgb)
    cv2.waitKey(0)


text_area(small1)
# text_area(large1)
# text_area(small2)
text_area(large2)