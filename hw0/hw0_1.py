import cv2 as cv
import numpy


img = cv.imread('image.png')
red_color = (0, 0, 255)
file = open('bounding_box.txt')

for Line in file.readlines():
	x1,y1,x2,y2 = map(int,Line.split())
	cv.rectangle(img, (x1,y1), (x2,y2), red_color, 3, cv.LINE_AA)

# cv.imwrite('hw0_110550098_1.png',img)
cv.imshow('frame',img)
cv.waitKey()