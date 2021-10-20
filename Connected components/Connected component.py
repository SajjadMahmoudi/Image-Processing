import cv2
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

"""----------------------------------------------------------------------"""

filename = "02.jpg"
img = cv2.imread(filename,0)
image = cv2.imread(filename,cv2.IMREAD_COLOR)
ret1,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


#imgarray = np.array(img)
#print(imgarray)

row = len(img2)
column = len(img2[0])

lable = 1
tmp = np.zeros((row, column))


"""Labling--------------------------------------------------------------"""

for x in range (row):
    for y in range (column):
        if img2[x, y] == 0:
            if tmp[x, y-1] !=0 or tmp[x-1, y] != 0 or tmp[x-1, y-1] != 0 or tmp[x-1, y+1] != 0:
               number = set()
               number.add(tmp[x-1, y])
               number.add(tmp[x, y-1])
               number.add(tmp[x-1, y-1])
               number.add(tmp[x-1, y+1])
               if list(sorted(number))[0] == 0:
                   tmp[x,y]= list(sorted(number))[1]
               else:
                    tmp[x,y]= list(sorted(number))[0]
            else:
                tmp[x, y] = lable
                lable += 1
"""--------------------------------------------------------------------"""

"""for x in range (row):
    str = []
    for y in range (column):
        str.append(tmp[x,y])
    print(str)
    str=[]"""

"""[x,y]
[x-1,y]
[x-1,y-1]
[x-1,y+1]
[x,y-1]
[x,y+1]
[x+1,y-1]
[x+1,y]
[x+1,y+1]"""

"""RELabling--------------------------------------------------------------"""
for x in range (row - 1, 0, -1):
    for y in range (column):
        if tmp[x, y] != 0:
             if tmp[x, y-1] !=0 or tmp[x+1, y] != 0 or tmp[x+1, y-1] != 0 or tmp[x+1, y+1] != 0:
                number = set()
                number.add(tmp[x, y-1])
                number.add(tmp[x+1, y])
                number.add(tmp[x+1, y-1])
                number.add(tmp[x+1, y+1])
                if list(sorted(number))[0] == 0:
                    tmp[x,y]= list(sorted(number))[1]
                else:
                    tmp[x,y]= list(sorted(number))[0]

for x in range (row ):
    for y in range (column - 1, 0, -1):
        if tmp[x, y] != 0:
              if tmp[x, y+1] !=0 or tmp[x-1, y] != 0 or tmp[x-1, y-1] != 0 or tmp[x-1, y+1] != 0:
               number = set()
               number.add(tmp[x-1, y])
               number.add(tmp[x, y+1])
               number.add(tmp[x-1, y-1])
               number.add(tmp[x-1, y+1])
               if list(sorted(number))[0] == 0:
                   tmp[x,y]= list(sorted(number))[1]
               else:
                    tmp[x,y]= list(sorted(number))[0]

for x in range (row - 1, 0, -1):
    for y in range (column):
        if tmp[x, y] != 0:
             if tmp[x, y-1] !=0 or tmp[x+1, y] != 0 or tmp[x+1, y-1] != 0 or tmp[x+1, y+1] != 0:
                number = set()
                number.add(tmp[x, y-1])
                number.add(tmp[x+1, y])
                number.add(tmp[x+1, y-1])
                number.add(tmp[x+1, y+1])
                if list(sorted(number))[0] == 0:
                    tmp[x,y]= list(sorted(number))[1]
                else:
                    tmp[x,y]= list(sorted(number))[0]                    

"""--------------------------------------------------------------------"""


"""image set--------------------------------------------------------------"""
imgset = set()

for x in range (row):
    for y in range (column):
        if tmp[x, y] != 0:
            imgset.add(tmp[x, y])
print(imgset)           
print(len(imgset))

"""--------------------------------------------------------------------"""

"""Color set------------------------------------------------------------"""
color = []
for i in range(len(imgset)):
    mycolor = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    color.append(mycolor)
"""Painting-------------------------------------------------------------"""

for x in range (row):
    for y in range (column):
        if tmp[x, y] != 0:
            for i in range(len(imgset)):
                if tmp[x, y] == list(imgset)[i]:
                    image[x,y] = color[i]

"""--------------------------------------------------------------------"""                
#plt.imshow(image)
#plt.show()

cv2.imshow("image",image)

cv2.waitKey(0)
cv2.destroyAllWindows()







