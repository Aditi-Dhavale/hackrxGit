import pytesseract
from PIL import Image
#However, too little text also has its faults. A webpage generally needs a minimum of 300 words for SEO purposes.
pytesseract.pytesseract.tesseract_cmd = r'C://Program Files/Tesseract-OCR/tesseract.exe' 
def rate_text_amount(filepath):
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image) #,config ='--psm 6'
    text_length = len(text)
    print(text_length)
    # You can define your own rating system here
    if text_length == 0:
        return "No text"
    elif text_length < 1200:
        return "Less than 300 words"
    else:
        return "More than 300 words"

filepath = 'C://aditi/competitions/hackerx4/contrastDataset/train/badContrast/3.jpg'
result = rate_text_amount(filepath)
print(result)