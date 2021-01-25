import numpy as np
import cv2
import math
from get_text_area import text_area
from text import preproccess, print_words
import pytesseract


def count_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = abs(math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180.0 / math.pi)
    return angle


if __name__ == "__main__":
    # Read image
    img = cv2.imread(r"images\books.jpg", cv2.IMREAD_COLOR)
    #img = cv2.resize(img, (670, 1000), interpolation=cv2.INTER_AREA)
    img_1 = cv2.pyrDown(img)
    img = cv2.pyrDown(img_1)

    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)

    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=300, maxLineGap=200)

    # Sort lines by x values
    new_list = sorted(lines, key=lambda line: line[0][0])

    # save only lines close to vertical
    new_list = [line for line in new_list if 70 <= count_angle(line) <= 90]

    # find indexes of new_list that will be excluded
    exclude = []
    for i in range(len(new_list) - 1):
        if abs(new_list[i][0][0] - new_list[i + 1][0][0]) < 35:
            exclude.append(i + 1)

    # exclude multiple lines that should be represented by a single line
    new_list = [i for j, i in enumerate(new_list) if j not in exclude]

    cropped_images = []

    # cutting single books from a shelf
    for i in range(len(new_list) - 1):
        x1 = new_list[i][0][0]
        x2 = new_list[i][0][2]
        x3 = new_list[i + 1][0][0]
        x4 = new_list[i + 1][0][2]
        crop_img = img_1[:, 2*min(x1, x2): 2*max(x3, x4)] # uzywam obrazka po 1 zmniejszeniu zamiast po dwoch na sam koniec zeby nie tracic jakosci
        cropped_images.append(crop_img)

    # saving images
    for i, image in enumerate(cropped_images):
        filename = "after_cropping/img" + str(i) + ".jpg"
        cv2.imwrite(filename, image)

    # putting lines on original image to see how algorithm worked
    for i, line in enumerate(new_list):
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Show result
    # cv2.imshow("Result Image", img)
    # cv2.waitKey(0)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    custom_config = r'--oem 3 --psm 6'
    for book in cropped_images:
        rotated = cv2.rotate(book, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        words = text_area(book, False, 8)
        words_from_rotated = text_area(rotated, True, 9)
        all_words = words + words_from_rotated
        for word in all_words:
            word = preproccess(word)
            print_words(word, custom_config)
        print('-------------------------------')

    # rotated = cv2.rotate(cropped_images[2], cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    # words = text_area(cropped_images[2], False, 1)
    # words_from_rotated = text_area(rotated, True, 2)


