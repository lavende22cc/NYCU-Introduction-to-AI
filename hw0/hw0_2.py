import cv2 as cv
import numpy as np

videoCapture = cv.VideoCapture()
videoCapture.open('video.mp4')

two_frame = []

for i in range(0,2):
	ret,frame = videoCapture.read()
	two_frame.append(frame);


diff = cv.absdiff(two_frame[0] , two_frame[1])
for i in diff:
	for j in i:
		j[0],j[2]=0,0

result = np.hstack((two_frame[0] , diff))
# cv.imwrite('hw0_110550098_2.png',result)
cv.namedWindow('frame',0)
cv.resizeWindow('frame',1080,360)
cv.imshow('frame',result)
cv.waitKey()