import numpy as np

def make_hidden_patch(a, fill):
    for y in range(0,3):
        for x in range(0,3):
            key = np.random.randint(2, size = 1)
            if key == 1 :
                for j in range(0,16) :
                    for i in range(0,16) :
                        a[i+x*16, j+y*16] = fill
    return a




