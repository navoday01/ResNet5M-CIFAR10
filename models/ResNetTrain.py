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
        1) BasicBlock/BottleNeck -> string
        2) Number of residual layers and blocks -> list<int>
        3) Number of channels for each residual block in residual layer -> list<int>
        4) Conv kernel size -> int
        5) Skip connection kernel size -> int
        6) Kernel size of Average pooling layer -> int
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 232, 268], 3, 1, 4)  # Reaches ~80% within 5 epochs

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

"""
TODO
1) Check Normalize values and fix them if incorrect. DONE
2) Try with higher precision float64 (Need CUDA). NOT NEEDED
2) Add code to continue training from pt file. NOT NEEDED 
3) Fix bugs for changing kernel size in conv layers and make it easy to change number of Residual Layers. NOT NEEDED
4) Add Tensor Board writer and viewer.
5) Add graphs of test/test loss vs time and train/test accuracy vs time.
6) Fix bugs in multithreading for dataloader. DONE
"""
