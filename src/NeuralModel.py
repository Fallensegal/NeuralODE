import torch
import torch.nn as NN
import numpy as np
from torchdiffeq import odeint_adjoint
import torch.nn.functional as F
"""
class ODEFunction(NN.Module):
    def __init__(self):
        super(ODEFunction, self).__init__()
        self.Linear1 = NN.Linear(784, 128)
        self.Linear2 = NN.Linear(128, 64)
        self.Relu = NN.ReLU()
    
    def forward(self, t, layer):
        layer = self.Relu(self.Linear1(layer))
        layer = self.Relu(self.Linear2(layer))
        return layer

class NeuralODE(NN.Module):
    def __init__(self, T=1.0):
        super(NeuralODE, self).__init__()
        self.ODE_FUNC = ODEFunction()
        self.T = T
        self.Linear3 = NN.Linear(64, 10)
    
    def forward(self, inputLayer):
        batchSize = inputLayer.shape[0]
        inputLayer_Flat = inputLayer.view(batchSize, -1)
        layer = odeint_adjoint(self.ODE_FUNC, inputLayer_Flat, torch.tensor([0.0, self.T]), method='dopri5')
        print(f"layer shape after odeint_adjoint: {layer.shape}")
        layer = layer[1].squeeze()
        print(f"layer shape after squeeze: {layer.shape}")
        output = self.Linear3(layer)
        print(f"output shape: {output.shape}")
        return output
"""
class ODEFunction(NN.Module):
    def __init__(self):
        super(ODEFunction, self).__init__()
        self.CONV1 = NN.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.POOL1 = NN.MaxPool2d(kernel_size=2)
        self.Relu = NN.ReLU()
    
    def forward(self, t, layer):
                    
        layer = self.Relu(self.CONV1(layer))
        layer = self.POOL1(layer)
        layer = layer.view(-1, 1, 28, 28)
        return layer
    
class NeuralODE(NN.Module):
    def __init__(self, T=1.0):        # Usually the T variable would be used for hidden layer, but in this instance we are initializing time
        super(NeuralODE, self).__init__()
        self.ODE_FUNC = ODEFunction()
        self.T = T
        self.ConvOutShape = self._GetConvOutShape()
        self.FullConn1 = NN.Linear(1*28*28, 10)
    
    def _GetConvOutShape(self):
        with torch.no_grad():
            layer = torch.zeros(1, 1, 28, 28)
            layer = self.ODE_FUNC(self.T, layer)
            return int(np.prod(layer.shape[1:]))

    def forward(self, inputLayer):
        batchSize = inputLayer.shape[0]
        inputLayer = inputLayer.view(batchSize, 1 , 28 , 28)
       
        layer = odeint_adjoint(self.ODE_FUNC, inputLayer, torch.tensor([0.0, self.T]), method='dopri5')[-1]
        output = self.FullConn1(layer)

        return F.softmax(output, dim=1)
