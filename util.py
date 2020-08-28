# List

from functools import reduce
import operator
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

# Math
def clamp(x,a,b):
    return min( max(a,x),b )

def normalize(X,a,b):
    #X = X.float()
    return (X-a)/(b-a)

# Images
import cv2
def cvimread(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def cvimwrite(path, im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im)


# Matplotlib
def imshow(x, figsize=(10,10)):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x)



# Time
from datetime import datetime
def timestamp():
    now = datetime.now()
    return now.strftime('%y%m%d_%H%M')

# FILE
import os
from os import path

def try_makedirs(directory, ensure_empty=False):

    if not path.exists(directory): 
        os.makedirs(directory)

    if ensure_empty:
        assert len(os.listdir(directory))==0, 'New directory needs to be empty'
