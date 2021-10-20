import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

"""------------------------------------------------------------------------------"""
filename = "1.png"
img = cv2.imread(filename, 0)

row = len(img)
column = len(img[0])

table = {}

for x in range(row):
    for y in range(column):
        if img[x, y] in table:
            table.update({img[x, y]: table[img[x, y]] + 1})
        else:
            table.update({img[x, y]: 1})

sortedtable = dict(sorted(table.items()))
# print(sortedtable)
# print(table)


plt.show()

graylevel = list(sortedtable.keys())
number = list(sortedtable.values())
# print(graylevel)
# print(number)

for i in range(1, len(number), 1):
    number[i] = number[i] + number[i - 1]

minnumber = number[0]
muxnumber = number[len(number) - 1]

for i in range(len(number)):
    number[i] = round((number[i] - minnumber) / muxnumber * 255)

for x in range(row):
    for y in range(column):
        for z in range(len(graylevel)):
            if img[x, y] == graylevel[z]:
                img[x, y] = number[z]
                break

# plt.bar(table.keys(), table.values(), align='center')
# plt.title('Histogram1')
# plt.bar(number, table.values(), align='center')
# plt.title('Histogram2')
cv2.imshow("image", img)
# plt.show()
