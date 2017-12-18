import sys
import numpy as np
from datetime import datetime

def python_sum(n):
    a = list(range(n))
    b = list(range(n))
    #in python3, range(n) returns a range object
    #rather than a list, so use list()
    c = []
    
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
        
    return c

def numpy_sum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b

    return c

# print(python_sum(5)) 
size = int(sys.argv[1])

start = datetime.now()
c = python_sum(size)
delta = datetime.now() - start
print('The last 2 elements of the python sum:',c[-2:],'python time:', delta.microseconds)

start = datetime.now()
c = numpy_sum(size)
delta = datetime.now() - start
print('The last 2 elements of the numpy sum:',c[-2:],'numpy time:', delta.microseconds)
