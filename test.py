import cv2
import numpy as np
import pytesseract
from PIL import Image

print("OpenCV version:")
print(cv2.__version__)

src_path = "/Users/victoriapetrov/Desktop/imageProject/"

def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #noise removal
    img = cv2.medianBlur(img,5)
    #img = cv2.GaussianBlur(img, (5, 5), 0) 
    cv2.imwrite(src_path + "blurred.png", img)


    #  Apply thresholding to get image with only black and white
    #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    cv2.imwrite(src_path + "threshold.png", img)

    kernel = np.ones((5,5),np.uint8)
    img = cv2.dilate(img, kernel, iterations = 1)
    img = cv2.erode(img, kernel, iterations = 1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(src_path + "opening.png", img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(Image.open(src_path + "opening.png"))

    return result

print ('--- Start recognize text from image ---')
print (get_string(src_path + "nl.png"))

print ("------ Done -------")
