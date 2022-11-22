import torch
from time import time
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchinfo import summary

class finalAcc:
    def __init__(self, model):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4913996458053589, 0.48215845227241516, 0.44653093814849854), (0.2470322549343109, 0.24348513782024384, 0.26158788800239563))
        ])
        self.model = model
        if torch.has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():  # Remove if you don't have MacBook
            self.device = "mps:0"
            print("Device set as MPS")
        elif torch.has_cuda and torch.cuda.is_available():
            self.device = "cuda:0"
            print("Device set as CUDA")
        else:
            self.device = "cpu"
            print("No GPU available using CPU")
        self.criterion = nn.CrossEntropyLoss()

    def getDataLoaders(self):
        trainset = datasets.CIFAR10(
            root = './data', train = True, download = True, transform = self.transform)
        testset = datasets.CIFAR10(
            root = './data', train = False, download = True, transform = self.transform)
        fullDataSet = torch.utils.data.ConcatDataset([trainset, testset])
        dataLoader = DataLoader(testset, batch_size = 100, shuffle = True, num_workers = 0)
        return dataLoader

    def test(self, dataLoader):
        self.model.eval()
        self.model.to(self.device)
        test_loss = 0
        correct = 0
        total = 0
        print("Starting final Test on all images in CIFAR-10 Test set")
        startTime = time()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataLoader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        endTime = time()
        print("Test Accuracy: " + str(100. * correct / total) + "% | Testing Time: " + str(endTime - startTime) + " sec")

    def finalTest(self):
        dataLoader = self.getDataLoaders()
        self.test(dataLoader)

if __name__ == "__main__":
    model = torch.jit.load('ResNet.pt')
    summary(model)
    acc = finalAcc(model)
    acc.finalTest()
