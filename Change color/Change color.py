import cv2
import numpy as np
import copy


image = cv2.imread('2.jpg')
image2 = cv2.imread('1.jpg')
copyimage = copy.copy(image)
row = len(image)
column = len(image[0])

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

down = np.array([10,50,0])
up = np.array([35,255,255])

mask  = cv2.inRange(hsv, down, up)
mask2 = cv2.bitwise_and(hsv, hsv, mask=mask)

RGBmask2 = cv2.cvtColor(mask2, cv2.COLOR_HSV2BGR)

for x in range(row):
    for y in range(column):
        if mask[x, y] == 255:
            copyimage[x, y] = (0,255,0)

for x in range(row):
    for y in range(column):
        if mask[x, y] == 255:
            image[x, y] = image2[x, y]
                    

cv2.imshow("image1", image)
cv2.imshow("image2", hsv)
cv2.imshow("image3", mask)
cv2.imshow("image4", copyimage)
cv2.imshow("image5", RGBmask2)


cv2.waitKey(0)
cv2.destroyAllWindows()
