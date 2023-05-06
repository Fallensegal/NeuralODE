
import torch
import torch.nn as NN
import torch.optim as Optim
import torchvision
import torchvision.transforms as transforms
from torchdiffeq import odeint_adjoint
from NeuralModel import *


def trainModel(Model, dataLoader, lossFunction, optimizer, device):
    Model.train()
    runningLoss = 0.0
    Correct = 0
    Total = 0

    for i, (inputs,labels) in enumerate(dataLoader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = Model(inputs)
        loss = lossFunction(outputs, labels)
        loss.backward()
        optimizer.step()

        # Loss Statistics
        runningLoss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        Total += labels.size(0)
        Correct += (predicted == labels).sum().item()

        avgLoss = runningLoss / (i+1)
        avgAcc = (Correct / Total) * 100
    
    return avgLoss, avgAcc

def testModel(Model, dataLoader, lossFunction, device):
    Model.eval()
    runningLoss = 0.0
    Correct = 0
    Total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataLoader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = Model(inputs)
            loss = lossFunction(outputs, labels)

            runningLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            Total += labels.size(0)
            Correct += predicted.eq(labels).sum().item()
    
    avgLoss = runningLoss / (i + 1)
    avgAcc = (Correct / Total) * 100

    return avgLoss, avgAcc



# Load the MNIST dataset
transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainSet = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=4)
testSet = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=False, num_workers=4)

# Initialize the NeuralODE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
function = ODE_Function(64)
ODE = NeuralODE(function)
NeuralMNIST_Model = NeuralMNISTClassifier(ODE).to(device)

lossFunction = NN.CrossEntropyLoss()
optimizer = Optim.Adam(NeuralMNIST_Model.parameters())

# Train the model
EPOCHS = 10
for epoch in range(EPOCHS):
    trainLoss, trainAcc = trainModel(NeuralMNIST_Model, trainLoader, lossFunction, optimizer, device)
    testLoss, testAcc = testModel(NeuralMNIST_Model, testLoader, lossFunction, device)
    print(f"Epoch: [{epoch + 1}/{EPOCHS}], Train Loss: {trainLoss:.4f}, Train Accuracy: {trainAcc:.2f}%, Test Loss: {testLoss:.4f}, Test Accuracy: {testAcc:.2f}%")

