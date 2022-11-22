from sys import exit
from ResNetModel import *
from torchinfo import summary
from TrainAndTest import *
from torch import optim
from PlotsAndGraphs import *


def main():
    # TENSORBOARD
    plots = PlotsAndGraphs()
    # MODEL
    print("Creating model...")
    """
    args:
        1) BasicBlock/BottleNeck -> list<string>
        2) Number of residual layers and blocks -> list<int>
        3) Number of channels for each residual block in residual layer -> list<int>
        4) Conv kernel size -> int **DO NOT CHANGE
        5) Skip connection kernel size -> int
        6) Kernel size of Average pooling layer -> int
    """
    model = ResNet([BasicBlock, BasicBlock, BasicBlock, BasicBlock], [2, 2, 2, 2], [64, 128, 232, 268], 3, 1, 4)  # Reaches ~80% within 5 epochs

    if summary(model).total_params > 5e+06:
        exit("Total Number of Parameters greater than 5 Million")

    print("Model created")

    # DEVICE
    print("Checking for GPU...")
    if torch.has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():  # Remove if you don't have MacBook
        device = "mps:0"
        print("Device set as MPS")
    elif torch.has_cuda and torch.cuda.is_available():
        device = "cuda:0"
        print("Device set as CUDA")
    else:
        device = "cpu"
        print("No GPU available using CPU")

    # DATALOADER
    print("Loading Datasets and creating Dataloader with 1 worker(s)...")
    datasets = TrainAndTest(plots)
    trainloader, testloader = datasets.getDataLoaders(batch_size = 128)
    print("Dataloader ready")

    # LOSS FUNCTION, OPTIMIZER AND SCHEDULER
    print("Loading Loss, Optimizer and Scheduler...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-04)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)

    # TRAINING, TESTING, SAVING
    print("Training and Saving model...")
    datasets.trainTestAndSave(200, model, device, criterion, optimizer, scheduler, trainloader, testloader)


if __name__ == "__main__":
    main()
