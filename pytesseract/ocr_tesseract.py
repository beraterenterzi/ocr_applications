import cv2
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

path = r'C:\Users\201311\Desktop\ocr\data\2.png'
img = cv2.imread(path)
config_tesseract = "--psm 6"

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tesseract_ocr(img, config_tesseract):
    text = pytesseract.image_to_string(img, lang='tur', config=config_tesseract)
    return text


# text = pytesseract.image_to_string(rgb, lang='tur', config="--psm 6")

# result = pytesseract.image_to_data(rgb)

# print(text)

full_text = ''
txt_file = 'result_ocr.txt'

for image in path:
    # print(image)
    img = cv2.imread(image)
    file_image = os.path.split(image)[-1]
    # print(file_image)
    file_image_separate = '================\n' + str(file_image)
    # print(file_image_separate)
    full_text = full_text + file_image_separate + '\n'

    text = tesseract_ocr(img, config_tesseract)
    # print(text)
    full_text = full_text + text
    print(full_text)

file_txt = open(txt_file, 'w+')
file_txt.write(full_text + '\n')
file_txt.close()
