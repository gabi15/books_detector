import cv2
import numpy as np


def text_area(image, is_horizontal, save_num):
    image_copy = image.copy()
    small = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.pyrDown(small)
    small = cv2.pyrDown(small)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    list_of_boxes = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.45 and w > 8 and h > 8:
            if is_horizontal:
                x1 = 4 * x
                x2 = 4*(x+w-1)
                y1 = 4 * y
                y2 = 4*(y+h-1)
            else:
                x1 = 2 * x
                x2 = 2*(x+w-1)
                y1 = 2 * y
                y2 = 2*(y+h-1)
            list_of_boxes.append(image_copy[y1:y2, x1:x2]) #using copy to avoid green frame
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i, img in enumerate(list_of_boxes):
        filename = "words/img" + str(save_num) + str(i) + ".jpg"
        cv2.imwrite(filename, img)

    cv2.imshow('rects', image)
    cv2.waitKey(0)
    return list_of_boxes

#normal = cv2.imread('after_cropping/img0.jpg') #pionowe - beda odczytywane wartosci mniejsze z grzbietu
#rotated = cv2.rotate(normal, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) #poziome bedzie odczytywany glowny tekst na grzbiecie, dlatego trzeba bedzie jeszcze zmniejszyc obrazek

# pion = text_area(normal, False, 7)
# poziom = text_area(rotated, True, 9)

