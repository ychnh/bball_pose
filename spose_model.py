import torch
import torchvision.models as models
from torch.nn import Linear,BatchNorm1d, ReLU, BatchNorm2d, Conv2d
from modeling.backbone import build_backbone

class spose_model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        r = 10
        L = 2*r+1
        self.compress = torch.nn.Conv2d(6,3,kernel_size=3, padding=1)
        #self.backbone = DeepLab(sync_bn=False, num_classes=10)
        #self.backbone = models.resnet18(num_classes=2*12)
        self.backbone = build_backbone('resnet', 16, BatchNorm2d)
        self.dimreduce = Conv2d(2048,16,kernel_size=3,padding=1)
        K1 = 16*16*16
        self.L1 = torch.nn.Sequential(Linear(K1,512), BatchNorm1d(512), ReLU(inplace=True) )
        self.L2 = torch.nn.Sequential(Linear(512,128), BatchNorm1d(128), ReLU(inplace=True) )
        self.L3 = torch.nn.Sequential(Linear(128,2*L), BatchNorm1d(2*L), ReLU(inplace=True) )
        self.softmax = torch.nn.Softmax(dim=2)

        self.K1= K1
        self.L = L

    def forward(self, A,B):
        x = torch.cat([A,B], dim=1)
        x = self.compress(x)
        x,_ = self.backbone(x)
        x = self.dimreduce(x).view(-1, self.K1)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x).view(-1, 2, self.L)
        x = self.softmax(x)
        return x

