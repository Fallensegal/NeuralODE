import torch
import torch.nn as NN
import numpy as np
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from torch import Tensor


def addTime(inFeatures, time):
    batchSize, channels, width, height = inFeatures.shape
    return torch.cat((inFeatures, time.expand(batchSize, 1, width, height)), dim=1)

class ODE_Function(NN.Module):
    def __init__(self, Dim):
        super(ODE_Function, self).__init__()
        self.conv1 = NN.Conv2d(Dim + 1, Dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = NN.BatchNorm2d(Dim)
        self.conv2 = NN.Conv2d(Dim + 1, Dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = NN.BatchNorm2d(Dim)
        self.relu = NN.ReLU()

    def forward(self, time, features):
        xt = addTime(features, time)
        layer = self.norm1(self.relu(self.conv1(xt)))
        layerT = addTime(layer, time)
        DXDT = self.norm2(self.relu(self.conv2(layerT)))
        
        return DXDT

class NeuralODE(NN.Module):
    def __init__(self, Function):
        super(NeuralODE, self).__init__()
        self.ODEFunc = Function
        
    def forward(self, Feature0, Time=Tensor([0., 1.]), returnWhole=False):
        Time = Time.to(Feature0)
        Feature = odeint_adjoint(self.ODEFunc, Feature0, Time, method='rk4')
        if returnWhole:
            return Feature
        else:
            return Feature[-1]
        
class NeuralMNISTClassifier(NN.Module):
    def __init__(self, ODEFunc):
        super(NeuralMNISTClassifier, self).__init__()
        self.Downsample = NN.Sequential(
            NN.Conv2d(1, 64, kernel_size=3, stride=1),
            NN.BatchNorm2d(64),
            NN.ReLU(inplace=True),
            NN.Conv2d(64, 64, kernel_size=4, stride=2),
            #NN.BatchNorm2d(64),
            #NN.ReLU(inplace=True),
            #NN.Conv2d(64, 64, kernel_size=4, stride=2),
            )
        self.ODEBlock = ODEFunc
        self.NORM = NN.BatchNorm2d(64)
        self.avgPool = NN.AdaptiveAvgPool2d((1, 1))
        self.FC = NN.Linear(64, 10)
        
    def forward(self, input):
        input = self.Downsample(input)
        input = self.ODEBlock(input)
        #input = self.NORM(input)
        input = self.avgPool(input)
        SHAPE = torch.prod(torch.tensor(input.shape[1:])).item()
        input = input.view(-1, SHAPE)
        OUTPUT = self.FC(input)
        return OUTPUT
