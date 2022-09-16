from easyocr import Reader
import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

lang_list = ['en', 'tr']
print(lang_list)
gpu = False

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


def easy_ocr(img):
    reader = Reader(lang_list, gpu)
    result = reader.readtext(img)
    return result


