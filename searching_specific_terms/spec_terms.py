import pytesseract
from pytesseract import Output
import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import spacy
from wordcloud import WordCloud
from spacy import displacy

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

directory_imgs = r"C:\Users\201311\Desktop\ocr\data"
paths = [os.path.join(directory_imgs, f) for f in os.listdir(directory_imgs)]
print(paths)


def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis("off")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()


for image in paths:
    image = cv2.imread(image, 1)
    show_img(image)

config_tesseract = "--psm 6"


def tesseract_ocr(img, config_tesseract):
    text = pytesseract.image_to_string(img, lang='tur', config=config_tesseract)
    return text


full_text = ''
txt_file = 'result.txt'

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

term_search = 'ödeme'

with open(r'C:\Users\201311\Desktop\ocr\data\results.txt') as f:
    results = [i.start() for i in re.finditer(term_search, f.read())]

print(results)

for image in paths:
    # print(image)
    img = cv2.imread(image)
    file_img = os.path.split(image)[-1]
    print('==================\n' + str(file_img))
    text = tesseract_ocr(img, config_tesseract)
    results = [i.start() for i in re.finditer(term_search, text)]
    print('Number of times the term {} appears: {}'.format(term_search, len(results)))
    print('\n')

### WORD CLOUD

nlp = spacy.load('tr')
print(spacy.lang.tr.stop_words.STOP_WORDS)
stop_words = spacy.lang.tr.stop_words.STOP_WORDS


def preprocessing(text):
    text = text.lower()

    document = nlp(text)
    tokens_list = []
    for token in document:
        # print(token)
        tokens_list.append(token.text)
    # print(tokens_list)

    tokens = [word for word in tokens_list if word not in stop_words]
    # print(tokens)
    tokens = ' '.join([str(element) for element in tokens])
    # print(tokens)
    return tokens


processed_full_text = preprocessing(full_text)

plt.figure(figsize=(20, 10))
plt.imshow(WordCloud().generate(full_text));  # of, what, the -> stopwords

plt.figure(figsize=(20, 10))
plt.imshow(WordCloud().generate(processed_full_text));

document = nlp(processed_full_text)

displacy.render(document, style='ent', jupyter=True)

for entity in document.ents:
    if entity.label_ == 'PER':
        print(entity.text, entity.label_)

font = '/content/calibri.ttf'


def write_text(text, x, y, img, font, color=(50, 50, 255), font_size=16):
    font = ImageFont.truetype(font, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - font_size), text, font=font, fill=color)
    img = np.array(img_pil)

    return img


def ocr_process_image(img, term_search, config_tesseract, min_conf):
    result = pytesseract.image_to_data(img, config=config_tesseract, lang='por', output_type=Output.DICT)
    number_of_times = 0
    for i in range(0, len(result['text'])):
        confidence = int(result['conf'][i])
        if confidence > min_conf:
            text = result['text'][i]
            if term_search.lower() in text.lower():
                x, y, img = box(i, result, img, (0, 0, 255))
                img = write_text(text, x, y, img, font, (50, 50, 225), 14)
                number_of_times += 1
    return img, number_of_times


term_search = 'ödeme'  # computer
for image in paths:
    # print(image)
    img = cv2.imread(image)
    img_original = img.copy()
    file_image = os.path.split(image)[-1]
    print('=================\n' + str(file_image))

    img, number_of_times = ocr_process_image(img, term_search, config_tesseract, min_conf)
    print('Number of times term {} appears in {}: {}'.format(term_search, file_image, number_of_times))
    print('\n')
    show_img(img)

term_search = 'ödeme'

os.makedirs('proceed_images', exist_ok=True)

for image in paths:
    # print(image)
    img = cv2.imread(image)
    img_original = img.copy()
    file_image = os.path.split(image)[-1]
    img, number_of_times = ocr_process_image(img, term_search, config_tesseract, min_conf)
    if number_of_times > 0:
        show_img(img)
        new_file_image = 'processed_' + file_image
        new_image = '/content/processed_images/' + str(new_file_image)
        cv2.imwrite(new_image, img)
