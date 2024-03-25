import numpy as np
import time
import test

def g(x, s2):
    return np.exp(-x**2/2/s2)

points = np.linspace(0,1,1000)

t1 = time.time()
for i in range(10000):
    r = test.gs(points, 1)
t2 = time.time()
print(r)

print(t2-t1)