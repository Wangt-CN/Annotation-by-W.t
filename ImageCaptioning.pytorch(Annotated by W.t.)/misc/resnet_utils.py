# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
        #定义网络结构,利用resnet里的层建立自己的网络
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0) #在第一位置增加一个维度,总共4个维度,代表一张图片

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0) #自适应平均池化 把最后一个卷基层池化到设定大小
        #相当于改了最后一部分fc层和att层
        return fc, att

