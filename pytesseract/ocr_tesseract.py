import cv2

import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

path = r'C:\Users\201311\Desktop\data\1.jpg'
img = cv2.imread(path)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(rgb, lang='tur', config="--psm 6")

result = pytesseract.image_to_data(rgb)

print(text)
