import random
from os.path import join
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from os import listdir
import numpy as np
from itertools import product
import cv2

def cvimread(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

class sphere_dataset(Dataset):

    def __init__(self, path, r):
        os.listdir(path)
        K = 50
        k1,k2 = 180/K, 360/K
        D = np.ndarray((K,K), dtype=np.object)

        for i,j in product(range(K), range(K)):
            D[i,j] = join(path, str(round(k1*i,1)) +'_' +str(round(k2*i,1)) + '.png')

        self.D, self.K, self.r = D,K,r

    def __getitem__(self, i):
        r,K = self.r, self.K
        L = 2*r+1

        p1,p2 = i%K, i//K

        n1,n2 = self.nhood(p1,r,K), self.nhood(p2,r,K)


        d1,d2 = random.sample(range(L),1)[0], random.sample(range(L),1)[0]
        #_p1,_p2 = random.sample(n1, 1)[0], random.sample(n2, 1)[0]
        _p1,_p2 = n1[d1], n2[d2]
        x = self.fmtx( cvimread(self.D[p1,p2]) )
        _x = self.fmtx( cvimread(self.D[_p1, _p2]) )

        #d1,d2 = (_p1-p1)%(L), (_p2-p2)%(L)

        y = torch.zeros(2,L)
        self.place_guassian(y[0], d1,L)
        self.place_guassian(y[1], d2,L)
        #y[0,d1] = 1
        #y[1,d2] = 1
        return {'x1':x, 'x2':_x, 'y':y, 'yidx':torch.LongTensor([d1,d2])}

    def place_guassian(self, arr, p,L):
        N = self.nhood(p,2,L)
        G = [1,4,6,4,1]
        G = [1/16*g for g in G]
        for i,g in zip(N,G):
            arr[i] = g

    def __len__(self):
        return self.K**2

    @staticmethod
    def fmtx(x):
        x = torch.from_numpy(x)
        x = x.permute(2,0,1).float()/255
        return x

    @staticmethod
    def idx_deg(deg,k):
        return deg//k

    def nhood(self, p, r, K):
        #need to find 2 indices left and right
        N = list(range(p-r, p+r+1))
        N = [n%(K) for n in N]
        return N

