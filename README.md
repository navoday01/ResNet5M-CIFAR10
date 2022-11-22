# ResNet5M-CIFAR10

The ResNet model has led to the establishment for the efficient training of deeper neural networks. This project describes how we achieved higher accuracy while using the same ResNet architecture and the methodology used to optimize the ResNet model with the constrain of 5 million trainable parameters on CIFAR-10 dataset.The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The hyperparameters that have the greatest influence on the model are also discussed.

![Alt text](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/assets/CIFAR10-2.png)

## ⚙️ Setup

1. Install all the requirements
```shell
pip3 install torch torchvision torchaudio torchinfo tensorboard matplotlib
```
2. Clone the GitHub repository
```shell
git clone https://github.com/navoday01/ResNet5M-CIFAR10.git
```
3. Change directory into folder
```shell
cd ResNet5M-CIFAR10
```


## 💽 Quick Start: using Google Colab

To run a demo file go to following google collab link:
`http://www.youtube.com`

## ⏳ Training
Run train script `train.py` to recreate similar model
```shell
python3 train.py
```
## 🖼️ Testing

 To Reproduce the accuracy of the model, run `test.py` and ensure the model is on the right folder. This script will normalise the images to right value.
```shell
python3 test.py
```


## 📊 Results
| Model Name        | Optimizer               | Total Parameters            | Test Accuracy               |
|-------------------|-------------------------|-----------------------------|-----------------------------|
| Mono              | 9GB                     | 12 hours                    |                             |       
| Stereo            | 6GB                     | 8 hours                     |                             |
| Mono + Stereo     | 11GB                    | 15 hours                    |                             |

## 📦 Conclusion

The final model under 5 Million parameters and employing SGD achieved an accuracy of 95.55% on the CIFAR-10 test data set by systematically adjusting the hyperpameters and optimizers.

## 👩‍⚖️ Acknowledgement

We would like to thank everyone whose comments and suggestions helped us with the project. We appreciate the constant assistance of Professors Chinmay Hegde, Arsalan Mosenia, and the teaching assistant Teal Witter. Last but not least, we would like to express our sincere gratitude to the teaching staff for providing us with the chance to complete these tasks and projects. They were highly beneficial and relevant to comprehending the ideas.


