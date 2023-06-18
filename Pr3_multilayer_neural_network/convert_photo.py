import cv2
from cv2 import *

massive = []
i = 1
data = []
for i, ii in enumerate(cv2.imread(f"../static/photo/{i}.png", 0)):
    for j, jj in enumerate(ii):
        if jj == 255:
            data.append(0)
        else:
            data.append(1)
massive.append(data)
print(massive)
