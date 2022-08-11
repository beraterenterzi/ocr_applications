import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

img = cv2.imread("C:/Users/201311/Downloads/drive-download-20220725T100439Z-001/Images/cup.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### THRESHOLD BINARY(belli bir threshold değeri üzerinde kalanları 1 altını 0 yapar)

# value, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

### THRESHOLD OTSU METHOD (optimal threshold değerini bulur ve uygular)

# value, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

### ADAPTIVE THRESHOLDING (yüzeyde farklı renkler varsa kullanılır)

# adaptive_avarage = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)

### GAUSSIAN ADAPTIVE THRESHOLDING (GAUSS DAĞILIMI ALARAK THRESHOLD BULUR)

# adaptive_gausian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)

### COLOR INVERSION

# invert = 255 - gray

### RESIZING

# increase = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
# decrease = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

### MORPHOLOGICAL OPERATIONS

## EROSION

# erosion = cv2.erode(gray, np.ones((5, 5), np.uint8))
#
# opening = cv2.dilate(erosion, np.ones((5, 5), np.uint8))

## DILATION

# dilation = cv2.dilate(gray, np.ones((5, 5), np.uint8))
#
# closing = cv2.erode(dilation, np.ones((5, 5), np.uint8))

### NOISE REMOVAL OPERATIONS

## AVARAGE BLUR

# average_blur = cv2.blur(gray, (5, 5))

## GAUSSIAN BLUR

# gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)

## MEADIAN BLUR

median_blur = cv2.medianBlur(gray, 3)

## BILATERAL BLUR

# bilateral_filter = cv2.bilateralFilter(gray, 15, 55, 45)

### TEXT DETECTION

config_tesseract = '--tessdata-dir tessdata'
text = pytesseract.image_to_string(median_blur, lang='en', config=config_tesseract)
print(text)

# print(gray)

cv2.imshow('ocr', text)
cv2.waitKey(0)
