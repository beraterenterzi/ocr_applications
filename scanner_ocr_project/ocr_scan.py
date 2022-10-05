import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
from re import A
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


img = cv2.imread(r"C:\Users\201311\Desktop\ocr\data\33.jpg")

original = img.copy()
show_img(img)
(H, W) = img.shape[:2]

print(H, W)

### Convert to gray scale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_img(gray)

### GAUSSIAN BLUR

blur = cv2.GaussianBlur(gray, (5, 5), 0)
show_img(blur)

### Border detection

# edged = cv2.Canny(blur, 60, 160)
# show_img(edged)


### Contours detection

def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)[:6]
    return conts


conts = find_contours(blur.copy())

for c in conts:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.05 * perimeter, True)
    if len(approximation) == 4:
        larger = approximation
        break

print(larger)

cv2.drawContours(img, larger, -1, (120, 255, 0), 28)
cv2.drawContours(img, [larger], -1, (120, 255, 0), 2)
show_img(img)


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


points_larger = sort_points(larger)
print(points_larger)

pts1 = np.float32(points_larger)
pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

transform = cv2.warpPerspective(original, matrix, (W, H))
show_img(transform)

text = pytesseract.image_to_string(transform, lang='tur', config='--psm 6')
print(text)

increase = cv2.resize(transform, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
show_img(increase)

text = pytesseract.image_to_string(increase, lang='tur', config='--psm 6')
print(text)

show_img(transform)

brightness = 50
contrast = 80
adjust = np.int16(transform)

adjust = adjust * (contrast / 127 + 1) - contrast + brightness
adjust = np.clip(adjust, 0, 255)
adjust = np.uint8(adjust)
show_img(adjust)

processed_img = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
show_img(processed_img)

margin = 18
img_edges = processed_img[margin:H - margin, margin:W - margin]
show_img(img_edges)

fig, im = plt.subplots(2, 2, figsize=(15, 12))
for x in range(2):
    for y in range(2):
        im[x][y].axis('off')
im[0][0].imshow(original)
im[0][1].imshow(img)
im[1][0].imshow(transform, cmap='gray')
im[1][1].imshow(img_edges, cmap='gray')
plt.show()

# def transform_image(image_file):
#     img = cv2.imread(image_file)
#     original = img.copy()
#     show_img(img)
#     (H, W) = img.shape[:2]
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (7, 7), 0)
#     edged = cv2.Canny(blur, 60, 160)
#     show_img(edged)
#     conts = find_contours(edged.copy())
#     for c in conts:
#         peri = cv2.arcLength(c, True)
#         aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#         if len(aprox) == 4:
#             larger = aprox
#             break
#
#     cv2.drawContours(img, larger, -1, (120, 255, 0), 28)
#     cv2.drawContours(img, [larger], -1, (120, 255, 0), 2)
#     show_img(img)
#
#     points_larger = sort_points(larger)
#     pts1 = np.float32(points_larger)
#     pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
#
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     transform = cv2.warpPerspective(original, matrix, (W, H))
#
#     show_img(transform)
#     return transform
#
#
# def process_img(img):
#
#     processed_img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
#     processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
#     return processed_img
