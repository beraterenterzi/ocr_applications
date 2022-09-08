import pytesseract
from pytesseract import Output
import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

directory_imgs = r"C:\Users\201311\Desktop\ocr\data"
paths = [os.path.join(directory_imgs, f) for f in os.listdir(directory_imgs)]
print(paths)


def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


for image in paths:
    image = cv2.imread(image, 1)
    show_img(image)

config_tesseract = "--tessdata-dir tessdata"


def tesseract_ocr(img, config_tesseract):
    text = pytesseract.image_to_string(img, lang='tur', config=config_tesseract)
    return text


full_text = ''
txt_file = 'result_ocr.txt'

for image in paths:
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

term_search = 'kredi'

with open('/results_ocr.txt') as f:
    results = [i.start() for i in re.finditer(term_search, f.read())]

print(results)
