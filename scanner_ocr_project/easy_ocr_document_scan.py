from easyocr import Reader
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt

lang_list = ['en', 'tr']
print(lang_list)
gpu = False


def img_show(img):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


img = cv2.imread(r'C:\Users\201311\Desktop\ocr\data\15.jpg')
original = img.copy()
img_show(img)
(H, W) = img.shape[:2]

reader = Reader(lang_list, gpu)
result = reader.readtext(img)

print(result)

font = r'C:\Users\201311\Documents\GitHub\ocr_applications\img_source\Fonts\calibri.ttf'


def write_text(text, x, y, img, font, color=(50, 50, 255), font_size=22):
    font = ImageFont.truetype(font, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - font_size), text, font=font, fill=color)
    img = np.array(img_pil)
    return img


def box_coordinates(box):
    (lt, rt, br, bl) = box
    lt = (int(lt[0]), int(lt[1]))
    rt = (int(rt[0]), int(rt[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    return lt, rt, br, bl


def draw_img(img, lt, br, color=(200, 255, 0), thickness=2):
    cv2.rectangle(img, lt, br, color, thickness)
    return img


def text_background(text, x, y, img, font, font_size=32, color=(200, 255, 0)):
    background = np.full((img.shape), (0, 0, 0), dtype=np.uint8)
    text_back = text_back = write_text(text, x, y, background, font, font_size=font_size)
    text_back = cv2.dilate(text_back, (np.ones((3, 5), np.uint8)))
    fx, fy, fw, fh = cv2.boundingRect(text_back[:, :, 2])
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, -1)
    return img


img = original.copy()

for (box, text, probability) in result:
    print(box, text, probability)
    lt, rt, br, bl = box_coordinates(box)
    img = draw_img(img, lt, br)
    img = write_text(text, lt[0], lt[1], img, font)
    # img = text_background(text, lt[0], lt[1], img, font, 18, (200, 255, 0))
    # img = write_text(text, lt[0], lt[1], img, font, (0, 0, 0), 18)

img_show(img)
