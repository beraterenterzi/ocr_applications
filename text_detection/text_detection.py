import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract

### PYTESSERACT

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
config_tesseract = "--tessdata-dir tessdata --psm 7"

### PREPROCESSING
detector = r'C:\Users\201311\Downloads\drive-download-20220725T100439Z-001\Models\frozen_east_text_detection.pb'
min_confidence = 0.9

width, height = 320, 320
image = r'C:\Users\201311\Desktop\data\13.jpg'

img = cv2.imread(image)

original = img.copy()

H = img.shape[0]
W = img.shape[1]
print(H, W)

proportion_W = W / float(width)
proportion_H = H / float(height)
print(proportion_W, proportion_H)

img = cv2.resize(img, (width, height))
H = img.shape[0]
W = img.shape[1]
print(H, W)

# cv2.imshow('ocr', img)
# cv2.waitKey(0)

### LOADING THE NEURAL NETWORK

layers_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

neural_network = cv2.dnn.readNet(detector)

blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), swapRB=True, crop=False)

neural_network.setInput(blob)

scores, geometry = neural_network.forward(layers_names)

rows, columns = scores.shape[2:4]

print(rows, columns)

boxes = []
confidences = []


### DECODING VALUES

def geometric_data(geometry, y):
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    angles_data = geometry[0, 4, y]

    return angles_data, xData0, xData1, xData2, xData3


def geometric_calculation(angles_data, xData0, xData1, xData2, xData3):
    (offsetX, offsetY) = (x * 4.0, y * 4.0)
    angle = angles_data[x]
    cos = np.cos(angle)
    sin = np.sin(angle)
    h = xData0[x] + xData2[x]
    w = xData1[x] + xData3[x]
    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
    beginX = int(endX - w)
    beginY = int(endY - h)

    return beginX, beginY, endX, endY


for y in range(0, rows):

    data_scores = scores[0, 0, y]
    angles_data, xData0, xData1, xData2, xData3 = geometric_data(geometry, y)

    for x in range(0, columns):
        if data_scores[x] < min_confidence:
            continue

        beginX, beginY, endX, endY = geometric_calculation(angles_data, xData0, xData1, xData2, xData3)
        confidences.append(data_scores[x])
        boxes.append((beginX, beginY, endX, endY))

detections = non_max_suppression(np.array(boxes), probs=confidences)
print(proportion_H, proportion_W)

img_copy = original.copy()
for (beginX, beginY, endX, endY) in detections:
    # print(beginX, beginY, endX, endY)
    beginX = int(beginX * proportion_W)
    beginY = int(beginY * proportion_H)
    endX = int(endX * proportion_W)
    endY = int(endY * proportion_H)

    # region of interest
    roi = img_copy[beginY:endY, beginX:endX]

    cv2.rectangle(original, (beginX, beginY), (endX, endY), (0, 255, 0), 2)

cv2.imshow('ocr', original)

roi = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('ocr2', roi)
img_copy = original.copy()

for (beginX, beginY, endX, endY) in detections:
    beginX = int(beginX * proportion_W)
    beginY = int(beginY * proportion_H)
    endX = int(endX * proportion_W)
    endY = int(endY * proportion_H)

    roi = img_copy[beginY:endY, beginX:endX]
    text = pytesseract.image_to_string(roi, lang='tur', config=config_tesseract)
    print(text)

    cv2.rectangle(original, (beginX, beginY), (endX, endY), (0, 255, 0), 2)

cv2.imshow('ocr3', roi)
cv2.waitKey(0)
