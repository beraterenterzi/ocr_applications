import pytesseract
from PIL import Image  # pip install Pillow

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# and start doing it

# your saved images on desktop
list_with_many_images = [
    r"C:\Users\201311\Documents\GitHub\ocr_applications\3.png"
]


# create a function that returns the text
def image_to_str(path):
    """ return a string from image """
    return pytesseract.image_to_string(Image.open(path))


# now pure action + csv part
with open("images_content_1.csv", "w+", encoding="utf-8") as file:
    file.write("ImagePath, ImageText")
    for image_path in list_with_many_images:
        text = image_to_str(image_path)
        print(text)
        line = f"{image_path}, {text}\n"
        file.write(line)
