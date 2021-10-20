from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse



img = cv.imread('1.jpg',0)

rows = len(img)
cols = len(img[0])

array1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
array2 = np.float32([[cols-1,0], [0,0], [cols-1,rows-1]])




warp_mat = cv.getAffineTransform(array1, array2)
warp_dst = cv.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))

while True:
    angle = int (input("Write angle: "))
    scale = float (input("Write scale: "))
    center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)

    rotate = cv.getRotationMatrix2D( center, angle, scale )
    warp_rotate = cv.warpAffine(warp_dst, rotate, (warp_dst.shape[1], warp_dst.shape[0]))
    

    cv.imshow('image', img)
    cv.imshow('Warp_Rotate', warp_rotate)

    
    cv.waitKey(0)
    cv.destroyAllWindows()
