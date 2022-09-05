import cv2
import numpy as np
import imutils
import pytesseract
from matplotlib import pyplot as plt


def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

img = cv2.imread(r'C:\Users\201311\Desktop\data\1.jpg')


def detect_plate(file_img):
    img = cv2.imread(file_img)
    (H, W) = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    show_img(edged)
    conts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)[:8]

    location = None
    for c in conts:
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
        if cv2.isContourConvex(aprox):
            if len(aprox) == 4:
                location = aprox
                break

    beginX = beginY = endX = endY = None
    if location is None:
        plate = False
    else:
        mask = np.zeros(gray.shape, np.uint8)

        img_plate = cv2.drawContours(mask, [location], 0, 255, -1)
        img_plate = cv2.bitwise_and(img, img, mask=mask)

        (y, x) = np.where(mask == 255)
        (beginX, beginY) = (np.min(x), np.min(y))
        (endX, endY) = (np.max(x), np.max(y))

        plate = gray[beginY:endY, beginX:endX]
        show_img(plate)

    return img, plate, beginX, beginY, endX, endY


def ocr_plate(plate):
    config_tesseract = "--psm 6"
    text = pytesseract.image_to_string(plate, config=config_tesseract)
    text = "".join(c for c in text if c.isalnum())
    return text


def recognize_plate(file_img):
    img, plate, beginX, beginY, endX, endY = detect_plate(file_img)

    if plate is False:
        print("It was not possible to detect!")
        return 0

    text = ocr_plate(plate)
    print(text)
    img = cv2.putText(img, text, (beginX, beginY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 255, 0), 2,
                      lineType=cv2.LINE_AA)
    img = cv2.rectangle(img, (beginX, beginY), (endX, endY), (150, 255, 0), 2)
    show_img(img)

    return img, plate


img, plate = recognize_plate(
    r'C:\Users\201311\Desktop\data\ocr.jpg')
