import torch
import torch.nn as NN
import torch.optim as Optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def trainModel(model, dataLoader, lossFunction, optimizer, device):
    model.train()
    runningLoss = 0.0
    Correct = 0
    Total = 0

    for i, (inputs,labels) in enumerate(dataLoader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
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


def testModel(model, dataLoader, lossFunction, device):
    model.eval()
    runningLoss = 0.0
    Correct = 0
    Total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataLoader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = lossFunction(outputs, labels)

            runningLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            Total += labels.size(0)
            Correct += predicted.eq(labels).sum().item()
    
            avgLoss = runningLoss / (i + 1)
            avgAcc = (Correct / Total) * 100
    
    return avgLoss, avgAcc 




# Load MNIST DataSet

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainSet = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testSet = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=4)
testLoader = DataLoader(testSet, batch_size=64, shuffle=False, num_workers=4)

# Load Pretrained Model
RESNET18 = models.resnet18(pretrained=True)

# Adjust for MNIST
RESNET18.conv1 = NN.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
numFTR = RESNET18.fc.in_features
RESNET18.fc = NN.Linear(numFTR, 10)

# Move to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESNET18 = RESNET18.to(device)

LossFunction = NN.CrossEntropyLoss()
Optimizer = Optim.Adam(RESNET18.parameters(), lr=0.001)

numEpochs = 10
for epoch in range(numEpochs):
    trainLoss, trainAcc = trainModel(RESNET18, trainLoader, LossFunction, Optimizer, device)
    testLoss, testAcc = testModel(RESNET18, testLoader, LossFunction, device)

    print(f"Epoch {epoch + 1}/{numEpochs}")
    print(f"Train Loss: {trainLoss:.4f}, Train Acc: {trainAcc:.4f}")
    print(f"Test Loss: {testLoss:.4f}, Test Acc:{testAcc:.4f}")