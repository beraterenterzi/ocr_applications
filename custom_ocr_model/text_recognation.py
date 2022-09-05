import numpy as np
import cv2
from keras.models import load_model
from imutils.contours import sort_contours
import imutils

network = load_model('/content/drive/MyDrive/Cursos - recursos/OCR with Python/Models/network')
network.summary()

img = cv2.imread('/content/test-manuscript01.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3, 3), 0)

adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)

invertion = 255 - adaptive

dilation = cv2.dilate(invertion, np.ones((3, 3)))

edges = cv2.Canny(dilation, 40, 150)

dilation = cv2.dilate(edges, np.ones((3, 3)))


def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sort_contours(conts, method='left-to-right')[0]
    return conts


conts = find_contours(dilation.copy())

min_w, max_w = 4, 160
min_h, max_h = 14, 140
img_copy = img.copy()
for c in conts:
    # print(c)
    (x, y, w, h) = cv2.boundingRect(c)
    # print(x, y, w, h)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
        roi = gray[y:y + h, x:x + w]
        # cv2_imshow(roi)
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 100, 0), 2)


## ROI extarction

def extract_roi(img):
    roi = img[y:y + h, x:x + w]
    return roi


## Thresholding

def thresholding(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh


## Resizing

def resize_img(img, w, h):
    if w > h:
        resized = imutils.resize(img, width=28)
    else:
        resized = imutils.resize(img, height=28)

    (h, w) = resized.shape
    dX = int(max(0, 28 - w) / 2.0)
    dY = int(max(0, 28 - h) / 2.0)

    filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    filled = cv2.resize(filled, (28, 28))
    return filled


## Normalization

def normalization(img):
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


characters = []


def process_box(gray, x, y, w, h):
    roi = extract_roi(gray)
    thresh = thresholding(roi)
    (h, w) = thresh.shape
    resized = resize_img(thresh, w, h)
    cv2.imshow(resized)
    normalized = normalization(resized)
    characters.append((normalized, (x, y, w, h)))


for c in conts:
    # print(c)
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
        process_box(gray, x, y, w, h)

pixels = np.array([pixel[0] for pixel in characters], dtype='float32')

digits = '0123456789'
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
characters_list = digits + letters
characters_list = [l for l in characters_list]

predictions = network.predict(pixels)

img_copy = img.copy()

for (prediction, (x, y, w, h)) in zip(predictions, boxes):
    i = np.argmax(prediction)
    # print(i)
    probability = prediction[i]
    # print(probability)
    character = characters_list[i]
    # print(character)

    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 100, 0), 2)
    cv2.putText(img_copy, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    print(character, ' -> ', probability * 100)

    cv2.imshow(img_copy)


def extract_roi(img, margin=2):
    roi = img[y - margin:y + h, x - margin:x + w + margin]
    return roi


conts = find_contours(dilation.copy())
characters = []

for c in conts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
        process_box(gray, x, y, w, h)

boxes = [b[1] for b in characters]
pixels = np.array([p[0] for p in characters], dtype='float32')
predictions = network.predict(pixels)

img_copy = img.copy()

for (prediction, (x, y, w, h)) in zip(predictions, boxes):
    i = np.argmax(prediction)
    probability = prediction[i]
    character = characters_list[i]

    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 100, 0), 2)
    cv2.putText(img_copy, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    print(character, ' -> ', probability * 100)

    cv2.imshow(img_copy)


def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 7)
    edges = cv2.Canny(blur, 40, 150)
    dilation = cv2.dilate(edges, np.ones((3, 3)))
    return gray, dilation


def prediction(predictions, characters_list):
    i = np.argmax(predictions)
    probability = predictions[i]
    character = characters_list[i]
    return i, probability, character


def draw_img(img_cp, character):
    cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 100, 0), 2)
    cv2.putText(img_cp, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

#### SOLUTION SOME PROBLEMS
