import torch
import easyocr
import os
import csv

reader = easyocr.Reader(['en', 'en'])
img_text = reader.readtext('3.png')
final_text = ""

for _, text, __ in img_text:  # _ = bounding box, text = text and __ = confident level
    final_text += " "
    final_text += text
print(final_text)


# def traverse(directory):
#     path, directory, files = next(os.walk(directory))
#     return files
#
#
# directory = r'C:\Users\201311\Documents\GitHub\ocr_applications'
# files_list = traverse(directory)
#
# images_text = {}
# for files in files_list:
#     img_text = reader.readtext(directory + '/' + files)
#     final_text = ""
#     for _, text, __ in img_text:
#         final_text += " "
#         final_text += text
#     images_text[files] = final_text
#
# keys = list(images_text.keys())
# new_keys = [int(k[4:-4]) for k in keys]
# new_keys.sort()

with open('image_easy_ocr.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow([final_text])
