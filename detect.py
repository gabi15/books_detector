import numpy as np
import cv2
import math
import pytesseract


def preproccess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1))
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.medianBlur(img, 5)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    sum = img[0][0]/255 + img[0][-1]/255 + img[-1][0]/255 + img[-1][-1]/255
    if sum < 2:
        img = np.invert(img)
    return img


def print_words(image, config):
    boxes = pytesseract.image_to_data(image, config=config)
    result = ''
    for b in boxes.splitlines()[1:]:
        b = b.split()
        if len(b) == 12 and len(b[11]) > 2:
            result += b[11] + ' '
    return result


def text_area(image):
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

        if r > 0.45 and w > 8 and h > 6:
            x1 = 4 * x
            x2 = 4*(x+w-1)
            y1 = 4 * y
            y2 = 4*(y+h-1)
            if h/w <1:
                list_of_boxes.append(image_copy[y1:y2, x1:x2])  # using copy to avoid green frame
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return list_of_boxes


def count_angle_lines(line):
    x1, y1, x2, y2 = line[0]
    angle = abs(math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180.0 / math.pi)
    return angle


def scale_input_photo(photo):
    h, w, c = photo.shape
    scale = 3000/h
    new_size = (int(w * scale), 3000)
    scaled_img = cv2.resize(photo, new_size)
    return scaled_img


def detect(filename):
    # Read image
    orginal_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = scale_input_photo(orginal_img)
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
    h_img, w_img, d = img.shape
    clear = np.ones([h_img, w_img])
    cv2.drawContours(clear, new_list, -1, (0, 0, 0), 3)
    clear = clear.astype('uint8')*255

    # Find the edges in the image using canny detector
    edges = cv2.Canny(clear, 50, 200)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=100, maxLineGap=20)
    new_list = sorted(lines, key=lambda line: line[0][0])
    new_list = [line for line in new_list if 70 <= count_angle_lines(line) <= 90]

    # make groups of lines that are really close (20 px) and picking the longest from the group
    LINE_DISTANCE = 20
    tab = [0]
    idx = 0
    for i in range(len(new_list) - 1):
        idx += 1
        if abs(new_list[i][0][0] - new_list[i + 1][0][0]) > LINE_DISTANCE:
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
        crop_img = img_1[:, 2 * min(x1, x2): 2 * max(x3, x4)]  # use image after first reduction instead of two because I don't want to loose quality
        cropped_images.append(crop_img)

    # saving images
    # for i, image in enumerate(cropped_images):
    #     filename = "after_cropping/img" + str(i) + ".jpg"
    #     cv2.imwrite(filename, image)

    # putting lines on original image to see how algorithm worked
    # for i, line in enumerate(new_list):
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # cv2.imshow("result",img)
    # cv2.waitKey(0)

    books_array = []

    for book in cropped_images:
        rotated = cv2.rotate(book, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        words_from_rotated = text_area(rotated)
        book_name=''
        for word in words_from_rotated:
            word = preproccess(word)
            book_name += print_words(word, custom_config) + ''
        if book_name != '':
            books_array.append(book_name)

    return books_array
