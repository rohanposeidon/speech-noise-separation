import numpy as np
op = np.load('processedinput.npy')
i = 0
n = len(op)
while i < n:
    print(float(i)/100)
    while op[i] == 0:
        i = i+1
    print(float(i)/100)
    print()
    print(float(i)/100)
    while op[i] == 1:
        i = i+1
    print(float(i)/100)
    print()