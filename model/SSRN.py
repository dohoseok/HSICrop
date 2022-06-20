import torch.nn as nn
import torch.nn.functional as F


class SPCModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SPCModuleIN, self).__init__()
                
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7,1,1), stride=(2,1,1), bias=False)
        #self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        
        input = input.unsqueeze(1)
        
        out = self.s1(input)
        
        return out.squeeze(1) 

class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(SPAModuleIN, self).__init__()
                
        # print('k=',k)
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,3,3), bias=False)
        #self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        # print(input.size())
        out = self.s1(input)
        out = out.squeeze(2)
        # print(out.size)
        
        return out

class ResSPC(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPC, self).__init__()
                
        self.spc1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm3d(in_channels),)
        
        self.spc2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),)
        
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
                
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))
        
        return F.leaky_relu(out + input)

class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPA, self).__init__()
                
        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm2d(in_channels),)
        
        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))
        
        return F.leaky_relu(out + input)

class SSRN(nn.Module):
    def __init__(self, num_classes=9, k=72):
        super(SSRN, self).__init__()

        self.layer1 = SPCModuleIN(1, 28)
        #self.bn1 = nn.BatchNorm3d(28)
        
        self.layer2 = ResSPC(28,28)
        
        self.layer3 = ResSPC(28,28)
        
        #self.layer31 = AKM(28, 28, [97,1,1])   
        self.layer4 = SPAModuleIN(28, 28, k=k)
        self.bn4 = nn.BatchNorm2d(28)
        
        self.layer5 = ResSPA(28, 28)
        self.layer6 = ResSPA(28, 28)

        self.fc = nn.Linear(28, num_classes)

    def forward(self, x):
        # print("[DHS][0] x", x.shape)

        x = F.leaky_relu(self.layer1(x)) #self.bn1(F.leaky_relu(self.layer1(x)))
        #print(x.size())
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer31(x)
        # print("[DHS][1] x", x.shape)

        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        
        return x