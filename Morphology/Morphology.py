import cv2
import numpy as np

image = cv2.imread('1.png')

row = len(image)
column = len(image[0])

kernel1 = np.ones((36,36),np.uint8)
kernel2 = np.ones((100,100),np.uint8)
kernel3 = np.ones((3,3),np.uint8)



closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)


edges = cv2.Canny(opening,100,200)
dilation = cv2.dilate(edges,kernel3,iterations = 1)

for x in range(row):
    for y in range(column):
        if dilation[x, y] == 255:
            image[x, y] = dilation[x, y]

            
cv2.imshow("image1", image)
cv2.imshow("closing", closing)
cv2.imshow("opening", opening)




cv2.waitKey(0)
cv2.destroyAllWindows()

