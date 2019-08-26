import imutils
import pytesseract
import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd= r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

img = cv2.imread('01.jpeg')
#image = imutils.resize(image, width=500)
cv2.imshow('realimage', img)
cv2.waitKey(0)

#from skimage import filters

#thre = filters.threshold_otsu(image)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayimage', grayscale)
cv2.waitKey(0)
gray = cv2.bilateralFilter(grayscale, 11, 17, 17)
cv2.imshow('2-bilateral Filter', gray)
cv2.waitKey(0)

edgedetect = cv2.Canny(gray, 100, 200)
cv2.imshow('edged', edgedetect)
cv2.waitKey(0)
contour, new = cv2.findContours(edgedetect.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img1 = img.copy()
cv2.drawContours(img, contour, 3, (0, 255, 0), 3)
#cv2.imshow('all contours', img1)
cv2.waitKey(0)
contour = sorted(contour, key = cv2.contourArea, reverse = True)[:30]
numberplateCnt = None
img2 = img.copy()
cv2.drawContours(img2, contour, -1, (0, 255, 0), 3)
cv2.imshow('contoursimg', img2)
cv2.waitKey(0)
count = 0
index = 7
for c in contour:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 *peri, True)
    if len(approx) == 4:
        NumberPlateCnt = approx
        x, y, w, h = cv2.boundingRect(c)
        new_img = img[y:y + h, x:x + w]
        cv2.imwrite('croppedImage' + str(index) + '.png', new_img)
        index+=1
        break
image = cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)
cv2.imshow('imageslast', image)
cv2.waitKey(0)
cropped_img_loc = 'croppedImage7.png'
cv2.imshow('a', cv2.imread(cropped_img_loc))
text = pytesseract.image_to_string(image, lang='eng')
print('number is=', text)
cv2.waitKey(0)



