import cv2

import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

path = 'C:/Users/201311/Documents/GitHub/ocr_applications/img_source/Images/cup.jpg'
img = cv2.imread(path)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(rgb)

result = pytesseract.image_to_data(rgb)


print(text)
