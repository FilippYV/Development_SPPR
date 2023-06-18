import random

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

# population = [[0],[ 2], [3], [4]]

# count_par = len(population) // 2
# share = []
# for i in range(count_par):
#     share.append(random.randint(1, len(population[0])))
# par = []
# for i in range(0, count_par * 2):
#     while len(par) != i + 1:
#         x = random.randint(0, len(population) - 1)
#         if x not in par:
#             par.append(x)
#
# new_par = []
# mas = []
# for i, ii in enumerate(par):
#     if i % 2 != 0 or i == 0:
#         mas.append(ii)
#     else:
#         new_par.append(mas)
#         mas = [ii]
# new_par.append(mas)
#
#
# print(par)
# print(new_par)
