import numpy as np
import cv2
import math
from get_text_area import text_area
from text import preproccess, print_words
import pytesseract

def count_angle_lines(line):
    x1, y1, x2, y2 = line[0]
    angle = abs(math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180.0 / math.pi)
    return angle

def scale_input_photo(photo):
    h, w, c = photo.shape
    scale = 3000/h
    newSize = (int(w * scale), 3000)
    scaledImg = cv2.resize(photo, newSize)
    return scaledImg


if __name__ == "__main__":
    # Read image
    orginalImg = cv2.imread(r"images\books.jpg", cv2.IMREAD_COLOR)
    img = scale_input_photo(orginalImg)
    #img = cv2.imread(r"images\books.jpg", cv2.IMREAD_COLOR)
    img_1 = cv2.pyrDown(img)
    img = cv2.pyrDown(img_1)
    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)

    # including pytesseract part
    pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'
    custom_config = r'--oem 3 --psm 6'

    new_contours = []
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #pick contours that are high
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 100:
            new_contours.append(contour)

    new_list = sorted(new_contours, key=lambda line: cv2.boundingRect(line)[0])

    # new white image with black contours
    hImg, wImg, d = img.shape
    clear = np.ones([hImg, wImg])
    cv2.drawContours(clear, new_list, -1, (0, 0, 0), 3)
    clear = clear.astype('uint8')*255
    # cv2.imshow('bbb', clear)
    # cv2.waitKey(0)

    # Find the edges in the image using canny detector
    edges = cv2.Canny(clear, 50, 200)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=100, maxLineGap=20)
    new_list = sorted(lines, key=lambda line: line[0][0])
    new_list = [line for line in new_list if 70 <= count_angle_lines(line) <= 90]

    # for i, line in enumerate(new_list):
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    #
    # cv2.imshow("result",img)
    # cv2.waitKey(0)

    # find indexes of new_list that will be excluded

    tab = [0]
    idx = 0
    for i in range(len(new_list) - 1):
        idx += 1
        if abs(new_list[i][0][0] - new_list[i + 1][0][0]) > 20:
            tab.append(idx)
    tab.append(len(new_list)-1)
    new_new_list = []
    for i in range(len(tab)-1):
        if tab[i] - tab[i+1] == 0:
            my_line = new_list[tab[i]]
        else:
            my_line = max(new_list[tab[i]:tab[i + 1]], key=lambda el: abs(el[0][1] - el[0][3]))
        new_new_list.append(my_line)

    # exclude multiple lines that should be represented by a single line
    new_list = new_new_list


    cropped_images = []

    # cutting single books from a shelf
    for i in range(len(new_list) - 1):
        x1 = new_list[i][0][0]
        x2 = new_list[i][0][2]
        x3 = new_list[i + 1][0][0]
        x4 = new_list[i + 1][0][2]
        crop_img = img_1[:, 2 * min(x1, x2): 2 * max(x3, x4)]  # uzywam obrazka po 1 zmniejszeniu zamiast po dwoch na sam koniec zeby nie tracic jakosci
        cropped_images.append(crop_img)

    print(cropped_images)
    # saving images
    # for i, image in enumerate(cropped_images):
    #     filename = "after_cropping/img" + str(i) + ".jpg"
    #     cv2.imwrite(filename, image)

    # putting lines on original image to see how algorithm worked
    for i, line in enumerate(new_list):
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow("result",img)
    cv2.waitKey(0)

    for book in cropped_images:
        rotated = cv2.rotate(book, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        words_from_rotated = text_area(rotated)
        for word in words_from_rotated:
            word = preproccess(word)
            print_words(word, custom_config)
        print('-------------------------------')
