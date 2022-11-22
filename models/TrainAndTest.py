from time import time
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class TrainAndTest():
    def __init__(self, plots):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4913996458053589, 0.48215845227241516, 0.44653093814849854), (0.2470322549343109, 0.24348513782024384, 0.26158788800239563))
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4913996458053589, 0.48215845227241516, 0.44653093814849854), (0.2470322549343109, 0.24348513782024384, 0.26158788800239563))
        ])
        self.best_acc = 0
        self.plots = plots

    def getDataLoaders(self, batch_size):
        trainset = datasets.CIFAR10(
            root = './data', train = True, download = True, transform = self.transform_train)
        testset = datasets.CIFAR10(
            root = './data', train = False, download = True, transform = self.transform_test)
        trainloader = DataLoader(
            trainset, batch_size = batch_size, shuffle = True, num_workers = 1)  # Change number of workers for multithreading
        testloader = DataLoader(
            testset, batch_size = 100, shuffle = False, num_workers = 0)
        return trainloader, testloader

    def train(self, epoch, model, device, criterion, optimizer, trainloader):
        model.to(device)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        startTime = time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.plots.graph(model, inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        endTime = time()
        print("Epoch: " + str(epoch) + " | Train Loss: " + str(train_loss/(batch_idx + 1)) + " | Train Accuracy: " + str(100.*correct/total) + "% | Training Time: " + str(endTime - startTime) + " sec")
        self.plots.plot("Train Loss", train_loss/(batch_idx + 1), epoch)
        self.plots.plot("Train Accuracy", 100.*correct/total, epoch)

    def test(self, epoch, model, device, criterion, testloader):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        startTime = time()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        endTime = time()
        acc = 100. * correct / total
        self.plots.plot("Test Loss", test_loss / (batch_idx + 1), epoch)
        self.plots.plot("Test Accuracy", 100. * correct / total, epoch)
        print("Epoch: " + str(epoch) + " | Test Loss: " + str(test_loss / (batch_idx + 1)) + " | Test Accuracy: " + str(100. * correct / total) + "% | Testing Time: " + str(endTime - startTime) + " sec")
        if acc > self.best_acc:
            model.cpu()
            model_scripted = torch.jit.script(model)
            model_scripted.save('ResNet.pt')
            print("Saved Model")
            self.best_acc = acc

    def trainTestAndSave(self, epochs, model, device, criterion, optimizer, scheduler, trainloader, testloader):
        for epoch in range(epochs):
            self.train(epoch, model, device, criterion, optimizer, trainloader)
            self.test(epoch, model, device, criterion, testloader)
            scheduler.step()