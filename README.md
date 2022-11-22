# ResNet5M-CIFAR10

The ResNet model has led to the establishment for the efficient training of deeper neural networks. This project describes how we achieved higher accuracy while using the same ResNet architecture and the methodology used to optimize the ResNet model with the constrain of 5 million trainable parameters on CIFAR-10 dataset.The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The hyperparameters that have the greatest influence on the model are also discussed.

![Alt text](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/assets/CIFAR10-2.png)

## ⚙️ Setup

1. Install all the requirements (PyTorch v1.13.0)
```shell
pip3 install torch torchvision torchaudio torchinfo tensorboard
```
2. Clone the GitHub repository
```shell
git clone https://github.com/navoday01/ResNet5M-CIFAR10.git
```
3. Change directory into folder
```shell
cd ResNet5M-CIFAR10
```


## 🏁 Quick Start: using Google Colab

To run a demo file go to following google collab link: [test model](https://colab.research.google.com/github/navoday01/ResNet5M-CIFAR10/blob/main/ResnetQuickTest.ipynb)

## ⏳ Training
Run train script `ResNetTrain.py` to recreate similar model
```shell
cd models
python3 ResNetTrain.py
```
## 🖼 Testing

 To Reproduce the accuracy of the model, run `FinalTesting.py` and **ensure the model is on the right folder and change the path in the file.** This script will normalise the images to right value.
```shell
cd models
python3 FinalTesting.py
```


## 📊 Results
| Sr. No.|    Model Name    |  # Conv Channels at layer  |  Optimizer  |  Params  |  Final Acc  |  File Link  |
|--------|------------------|----------------------------|------------ |----------|-------------|-------------|
|   1    |                  |     [64,128,192,256]       |    SGD+M    |   4.47M  |  93.32%     |             |
|   2    |                  |     [64,128,192,256]       |    Adagrad  |   4.47M  |  90.55%     |             |
|   3    |                  |     [64,128,192,256]       |    RMSProp  |   4.47M  |  89.13%     |             |
|   4    |                  |     [64,128,192,256]       |    Adam     |   4.47M  |  93.05%     |             |
|   5    |                  |     [64, 128, 232, 268]    |    SGD+M    |   4.99M  |  95.55%     |             |
|   6    |                  |     [64, 128, 232, 268]    |    SGD+M    |   4.99M  |  95.55%     |             |

## 📦 Conclusion

The final model under 5 Million parameters and employing SGD achieved an accuracy of 95.55% on the CIFAR-10 test data set by systematically adjusting the hyperpameters and optimizers.

## 👩‍⚖️ Acknowledgement

We would like to thank everyone whose comments and suggestions helped us with the project. We appreciate the constant assistance of Professors Chinmay Hegde, Arsalan Mosenia, and the teaching assistant Teal Witter. Last but not least, we would like to express our sincere gratitude to the teaching staff for providing us with the chance to complete these tasks and projects. They were highly beneficial and relevant to comprehending the ideas.


