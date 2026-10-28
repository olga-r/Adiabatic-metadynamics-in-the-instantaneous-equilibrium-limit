import numpy as np

def periodic(x, left, right):
    arg = np.pi*(x-left)/(right-left)
    return 5*(np.cos(2*arg))**2+(np.cos(arg))**2-0.4875
