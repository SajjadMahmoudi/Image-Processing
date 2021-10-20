import cv2
import numpy as np

img1 = cv2.imread("1.jpeg", 0)
img2 = cv2.imread("2.jpeg", 0)

# sift Detector
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

    
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

    
matching_result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:80], None, flags=2)

#cv2.imshow("Img1", img1)
#cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
