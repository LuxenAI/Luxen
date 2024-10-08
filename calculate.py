# start of code

# this is by NO MEANS finished, it is just the starting code for calculating size and color stuff

import numpy as np
import pandas as pd
import random

x = random_int = random.randint(0, 255)
y = random_int = random.randint(0, 255)
z = random_int = random.randint(0, 255)
RGB = [x, y, z]

a = random_int = random.randint(0, 30)
b = random_int = random.randint(0, 30)
c = random_int = random.randint(0, 30)
size = [a, b, c]

average = (x + y + z) / len(RGB)
print(average)

avg_size = (size[0] + size[1] + size[2]) / len(size)
print(avg_size)

if average > 100:
    if avg_size > 20:
        print("Still have a defeciency")
    elif 10 < avg_size < 20:
        print("The object is a sphere")
    else:
        print("The object is a cube")
if average < 100:
    if avg_size > 20:
        print("The object is a cube")
    elif 10 < avg_size < 20:
        print("The object is a sphere")
    else:
        print("Still have a defeciency")



