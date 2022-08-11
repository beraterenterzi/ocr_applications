import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

detector = r'C:\Users\berat\Downloads\drive-download-20220725T100439Z-001\Models\frozen_east_text_detection.pb'

width, height = 320, 320
image = r'C:\Users\berat\Downloads\drive-download-20220725T100439Z-001\Images\cup.jpg'

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

cv2.imshow('ocr', img)
cv2.waitKey(0)
