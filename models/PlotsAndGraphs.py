import torchvision
from torch.utils.tensorboard import SummaryWriter

class PlotsAndGraphs:
    def __init__(self):
        self.writer = SummaryWriter('runs/ResNet')

    def plotRandomImg(self, dataLoader):
        dataiter = iter(dataLoader)
        images, labels = next(dataiter)
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image("Images of CIFAR-10", img_grid)

    def graph(self, model, image):
        self.writer.add_graph(model, image)

    def plot(self, name, value, epoch):
        self.writer.add_scalar(str(name), value, epoch)