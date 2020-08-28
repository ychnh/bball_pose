import argparse
from os import path
import torch
from torch.utils.data import DataLoader
from spose_model import spose_model
from dataset import sphere_dataset
from matplotlib.pyplot import imshow
import util

util.try_makedirs('save')
dataset = sphere_dataset('data', 10)
model = spose_model().cuda()

device = 1
epochs = 25
batch_size = 8

trn_dtldr = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

from torch import diag, matmul
def F(a,b):
    return diag( matmul(a,b.T) ).sum()
def loss(py,y):
    py1,y1 = py[:, 0], y[:, 0]
    py2,y2 = py[:, 1], y[:, 1]
    #l1 = torch.bmm(py1,y1).sum() + torch.bmm(py1,py1)
    #l2 = torch.bmm(py2,y2).sum() + torch.bmm(py2,py2)
    l1 = -F(py1,y1)# + F(py1,py1)
    l2 = -F(py2,y2)# + F(py2,py2)
    return l1+l2
#loss = torch.nn.L1Loss().cuda()
#loss = torch.nn.MSELoss().cuda()
#loss = torch.nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam(model.parameters())

E,BT = epochs, len(trn_dtldr)
for e in range(E):
    for b, batch in enumerate(trn_dtldr):
        A,B = batch['x1'].cuda(), batch['x2'].cuda()
        y = batch['y'].cuda()
        #y = batch['yidx'].cuda()
        py = model(A,B)
        optimizer.zero_grad()
        l = loss(py,y)
        l.backward()
        optimizer.step()
        if b%(BT//10)==0: print('Epoch',e,'Batch',b,'Loss',l.item())

    torch.cuda.empty_cache()
    if (e+1)%5==0: torch.save( model.state_dict(), 'save/model'+str(e)+'.pth')

