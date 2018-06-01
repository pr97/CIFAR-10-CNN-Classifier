import numpy as np

m = np.arange(8).reshape((2,2,2))
print(m)
n = np.rot90(m, 1, (1,2))
print(n)