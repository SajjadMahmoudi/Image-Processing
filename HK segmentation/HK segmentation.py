import numpy as np
import cv2
import math  


filename = "3.jpg"
image = cv2.imread(filename)
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

row = len(img)
column = len(img[0])

E = np.zeros((row, column))
F = np.zeros((row, column))
G = np.zeros((row, column))
e = np.zeros((row, column))
f = np.zeros((row, column))
g = np.zeros((row, column))
N = np.zeros((row, column))
H = np.zeros((row, column))
K1 = np.zeros((row, column))
K2 = np.zeros((row, column))
K = np.zeros((row, column))




IS = cv2.GaussianBlur(img,(3,3),0)
#first division 
X = cv2.Sobel(IS, cv2.CV_64F, 1, 0)
Y = cv2.Sobel(IS, cv2.CV_64F, 0, 1)
XY = cv2.Laplacian(IS, cv2.CV_64F)
#second division 
X2 = cv2.Sobel(X, cv2.CV_64F, 1, 0)
Y2 = cv2.Sobel(Y, cv2.CV_64F, 0, 1)
XY2 = cv2.Laplacian(XY, cv2.CV_64F)

for x in range(row):
    for y in range(column):
        E[x,y] = 1+X[x,y]**2

for x in range(row):
    for y in range(column):
        G[x,y] = 1+Y[x,y]**2

for x in range(row):
    for y in range(column):
        F[x,y] = Y[x,y]*X[x,y]

for x in range(row):
    for y in range(column):
        N[x,y] = E[x,y]+G[x,y]

for x in range(row):
    for y in range(column):
        N[x,y] = math.sqrt(N[x,y])

for x in range(row):
    for y in range(column):
        e[x,y] = X2[x,y]/N[x,y]

for x in range(row):
    for y in range(column):
        f[x,y] = Y2[x,y]/N[x,y]

for x in range(row):
    for y in range(column):
        g[x,y] = XY2[x,y]/N[x,y]

for x in range(row):
    for y in range(column):
        H[x,y] = -(e[x,y]*G[x,y]-2*f[x,y]*F[x,y]+g[x,y]*E[x,y])

for x in range(row):
    for y in range(column):
        K1[x,y] = (e[x,y]*g[x,y]-f[x,y]**2)
for x in range(row):
    for y in range(column):
        K2[x,y] = (E[x,y]*G[x,y]-F[x,y]**2)
for x in range(row):
    for y in range(column):
        K[x,y] = K1[x,y]/K2[x,y]

for x in range(row):
    for y in range(column):
        if H[x,y] == 0 and K[x,y] == 0:
            image [x,y] = [0,0,0]
        elif H[x,y] == 0 and K[x,y] > 0:
            image [x,y] = [255,255,0]
        elif H[x,y] == 0 and K[x,y] < 0:
            image [x,y] = [0,255,0]
        elif H[x,y] > 0 and K[x,y] > 0:
            image [x,y] = [255,0,255]
        elif H[x,y] > 0 and K[x,y] < 0:
            image [x,y] = [255,0,0]
        elif H[x,y] < 0:
            image [x,y] = [0,0,255]



cv2.imshow("img",img)
cv2.imshow("image",image)


cv2.waitKey(0)
cv2.destroyAllWindows()
