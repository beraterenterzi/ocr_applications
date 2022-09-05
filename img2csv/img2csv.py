import pyautogui
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

text = pytesseract.image_to_string(r'C:\Users\201311\Desktop\data\1.jpg', config="--psm 6")

f = open(r'C:\Users\berat\Documents\GitHub\ocr_applications\5.csv', 'w')
f.write(text)
f.close()
print(text)
