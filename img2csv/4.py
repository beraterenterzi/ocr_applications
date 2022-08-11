import cv2
import pytesseract
import xlsxwriter
import csv

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

img = cv2.imread(r"C:\Users\berat\Documents\GitHub\ocr_applications\3.png")

workbook = xlsxwriter.Workbook('result.xlsx')
worksheet = workbook.add_worksheet()

row = 0
col = 0

part1 = []
part2 = []

# Get the size
(h, w) = img.shape[:2]

# Initialize indexes
increase = int(w / 2)
start = 0
end = start + increase

# For each part
for i in range(0, 2):

    # Get the current part
    cropped = img[0:h, start:end]

    # Convert to the gray-scale
    gry = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Threshold
    thr = cv2.threshold(gry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR
    txt = pytesseract.image_to_string(thr, config="--psm 6")
    print(txt)

    # Add ocr to the corresponding part
    txt = txt.split("\n")

    if i == 0:
        for sentence in txt:
            part1.append(sentence)
    else:
        for sentence in txt:
            part2.append(sentence)

    # Set indexes
    start = end
    end = start + increase

for txt1, txt2 in zip(part1, part2):
    worksheet.write(row, col, txt1)
    worksheet.write(row, col + 1, txt2)
    row += 1

workbook.close()

