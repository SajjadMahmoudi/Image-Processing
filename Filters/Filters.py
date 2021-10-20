import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

filename = "1.jpg"
img = cv2.imread(filename, 0)
row = len(img)
column = len(img[0])
"""**************************************************************************"""


def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))
    return g / g.sum()


"""**************************************************************************"""

gaussian_kernel_array = gaussian_kernel(2)
# print(gaussian_kernel_array)

img2 = cv2.filter2D(img, -1, gaussian_kernel_array)

"""plt.imshow(gaussian_kernel_array, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()
plt.show()"""
H = np.zeros((row, column))
R = 80
for i in range(row):
    for j in range(column):
        if ((i - row / 2 - 1) * (i - row / 2 - 1) + (j - column / 2 - 1) * (j - column / 2 - 1)) < R * R:
            H[i, j] = 1

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
magnitude_spectrum = np.asarray(magnitude_spectrum, np.uint8)

F = fshift * H

f_ishift = np.fft.ifftshift(F)
img_back = np.fft.ifft2(f_ishift)
MS = 20 * np.log(np.abs(img_back))
MS = np.asarray(img_back, np.uint8)

cv2.imshow("1", img)
cv2.imshow("2", magnitude_spectrum)
cv2.imshow("3", MS)
cv2.imshow("4", img2)


cv2.waitKey(0)
cv2.destroyAllWindows()
