
import numpy as np
from numpy import linalg as LA

if __name__ == '__main__':

    a = np.array([3., 4.])
    b = LA.norm(a)
    print(b)


