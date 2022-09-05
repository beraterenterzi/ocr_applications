import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt


def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# img = cv2.imread(r'C:\Users\201311\Desktop\data\2.jpg')
img = cv2.imread(r'C:\Users\201311\Documents\GitHub\ocr_applications\img_source\Images\Images Project 2\book1.jpg')
original = img.copy()
show_img(img)
(H, W) = img.shape[:2]
print(H, W)


def find_contours(img):

    conts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)[:6]
    return conts


def sort_points(points):

    points = points.reshape((4, 2))
    # print(points.shape)
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    # print(new_points.shape)
    # print(new_points)
    add = points.sum(1)
    # print(add)

    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]
    dif = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(dif)]
    new_points[3] = points[np.argmax(dif)]
    # print(new_points)

    return new_points


def transform_image(image_file):

    img = cv2.imread(image_file)
    original = img.copy()
    show_img(img)
    (H, W) = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 60, 160)
    show_img(edged)
    conts = find_contours(edged.copy())
    for c in conts:
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(aprox) == 4:
            larger = aprox
            break

    cv2.drawContours(img, larger, -1, (120, 255, 0), 28)
    cv2.drawContours(img, [larger], -1, (120, 255, 0), 2)
    show_img(img)

    points_larger = sort_points(larger)
    pts1 = np.float32(points_larger)
    pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transform = cv2.warpPerspective(original, matrix, (W, H))

    show_img(transform)
    return transform


def process_img(img):
    processed_img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    return processed_img


img = transform_image(r'C:\Users\201311\Desktop\data\scanned.jpg')
img = process_img(img)
show_img(img)
