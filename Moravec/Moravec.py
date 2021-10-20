import cv2
import numpy as np
import copy
import math    



filename = "1.jpg"
colorimg = cv2.imread(filename)
im = cv2.cvtColor(colorimg, cv2.COLOR_BGR2GRAY)


"""---------------------------------------------------------------------"""
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
"""---------------------------------------------------------------------"""

img = im2double(im)


row = len(img)
column = len(img[0])

"""--------------------------ROI-----------------------------------------"""
R=15  
C=15

cR = math.floor(row/R)
cC = math.floor(column/C)

"""---------------------------------------------------------------------"""


kernel = np.ones((5, 5), np.uint8)
tmp = np.zeros((row, column))
tmp2 = np.zeros((row, column))



I1 = np.array([-1, 1])
I2 = np.array([[-1], [1]])
I3 = np.array([[-1,0], [0,1]])
I4 = np.array([[0,-1], [1,0]])


img1 = cv2.filter2D(img , -1 , I1)
img2 = cv2.filter2D(img , -1 , I2)
img3 = cv2.filter2D(img , -1 , I3)
img4 = cv2.filter2D(img , -1 , I4)

img1 = img1**2
img2 = img2**2
img3 = img3**2
img4 = img4**2

img12 = cv2.filter2D(img1 , -1 , kernel)
img22 = cv2.filter2D(img2 , -1 , kernel)
img32 = cv2.filter2D(img3 , -1 , kernel)
img42 = cv2.filter2D(img4 , -1 , kernel)


for x in range(row):
    for y in range(column):
        tmp[x,y] = min(img12[x,y],img22[x,y],img32[x,y],img42[x,y])

"""----------------------------ROI--------------------------------------"""
def maximume(roi,i,j):
    Max = roi[0,0]
    Xmax = 0
    Ymax = 0

    for x in range (len(roi)):
        for y in range (len(roi[0])):
          if roi[x,y]> Max:
              Max = roi[x,y]
              Xmax = x
              Ymax = y
    if Max > 0.15 :
        tmp2[i*cR + Xmax, j*cC + Ymax] = 1
    #print(Xmax,Ymax)
"""---------------------------------------------------------------------"""


for i in range (R):
    for j in range (C):
        if i == R - 1 and j == C - 1:
            ROI = tmp[i*cR:row, j*cC:column]
        elif i == R - 1:
            ROI = tmp[i*cR:row, j*cC:(j+1)*cC]
        elif j == C - 1:
            ROI = tmp[i*cR:(i+1)*cR, j*cC:column]
        else:
            ROI = tmp[i*cR:(i+1)*cR, j*cC:(j+1)*cC]

        #print(ROI,len(ROI),len(ROI[0]))
        maximume(ROI,i,j)

"""---------------------------------------------------------------------"""



"""---------------------------------Highlighting------------------------"""

for x in range(row):
    for y in range(column):
        if tmp2[x, y] == 1:
                    
            colorimg = cv2.circle(colorimg, (y,x), 2 , (0,0,255), -1)

"""----------------------------------------------------------------------"""


cv2.imshow("Final image", colorimg)

#cv2.imshow("image5", tmp)
#cv2.imshow("image6", tmp2)
#cv2.imshow("image1", img12)
#cv2.imshow("image2", img22)
#cv2.imshow("image3", img32)
#cv2.imshow("image4", img42)


cv2.waitKey(0)
cv2.destroyAllWindows()

