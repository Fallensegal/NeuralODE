
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchdiffeq import odeint_adjoint

class ODEFunction(nn.Module):
    def __init__(self):
        super(ODEFunction, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

class NeuralODE(nn.Module):
    def __init__(self, T=1.0):
        super(NeuralODE, self).__init__()
        self.ode_func = ODEFunction()
        self.T = T
        self.fc = nn.Linear(64 * 14 * 14, 10)

    def forward(self, x):
        x = odeint_adjoint(self.ode_func, x, torch.tensor([0.0, self.T]), method='dopri5')[-1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Initialize the NeuralODE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralODE().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

print("Finished Training")

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


"""
import torch
import torch.nn as NN
import torch.optim as Optim
import torchvision
import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
from NeuralModel import *


def MNIST_Accuracy(model, testLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    with open("MNIST_Accuracy.txt", "a") as f:
        f.write(f"Accuracy: {accuracy:.2f}%\n")
    
    return (correct / total) * 100

#############################################################################################################

# Loading MNIST Dataset
MNIST_Transform = Transforms.Compose([
    Transforms.ToTensor(),
    Transforms.Normalize((0.1307,), (0.3081,))  # Mean and STDEV is Standard MNIST Parameters
])

TrainSet = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=MNIST_Transform)
TestSet = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=MNIST_Transform)
TrainLoader = DataLoader(TrainSet, batch_size=64, shuffle=True, num_workers=4)
TestLoader = DataLoader(TestSet, batch_size=64, shuffle=False, num_workers=4)

# Train Defined Model
NeuralODE_Model = NeuralODE()
LossFunction = NN.CrossEntropyLoss()
Optimizer = Optim.Adam(NeuralODE_Model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NeuralODE_Model = NeuralODE_Model.to(device)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    NeuralODE_Model.train()
    for i, (images, labels) in enumerate(TrainLoader):
        images, labels = images.to(device), labels.to(device)
        Optimizer.zero_grad()

        outputs = NeuralODE_Model(images)
        loss = LossFunction(outputs, labels)
        loss.backward()
        Optimizer.step()

        NeuralODE_Model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in TestLoader:
                images, labels = images.to(device), labels.to(device)
                outputs = NeuralODE_Model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, EPOCHS, i + 1, len(TrainLoader), loss.item(), (correct / total) * 100))

# Predict Testing Accuracy
MNIST_Accuracy(NeuralODE_Model, TestLoader)
"""