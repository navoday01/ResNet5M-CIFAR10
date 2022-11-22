# ResNet5M-CIFAR10

## About

## Methedology

## Results

### Convolution Channel

The number of channels is the decisive factor in attaining high accuracy on the CIFAR-10's test data set. After careful evaluation, it was determined that having higher channels at each layer guarantees a good performance. On one such occasion, a model trained using a lower number of channels had an average accuracy on the test data set of less than 90%. It was noted that a channel size greater than 200 at the final layer dramatically increases accuracy and guarantees it greater than 90%.

### Residual Blocks

#### Number of Residual Block

The CIFAR-10 data set has lower resolution images and having deeper networks did not increase accuracy on the test data set but instead had the opposite effect. Training attempts were made using ResNet 56 and 101 architectures. By carefully reducing the number of channels in each Convolutional to ensure the total number of parameters did not exceed 5 Million, these attempts were fruitless as the average accuracy in these cases always fell short of the 95% which was aimed.

#### Residual Block Style

Each layer in the model was tested using the standard Residual Block, Bottleneck Block or a combination of both at each layer. After evaluation of each style and combination of both, it can be safely said that the standard Residual Block outperforms the rest. Moreover, it also uses fewer parameters, so this ended up being the favorite. It was stated by the authors that the Bottleneck Block works best on deeper networks and since our input images being lower resolution and restrictions of $5$ Million parameters causes some compatibility issues with the Bottleneck style.

### Optimizers
Optimizers are critical to reducing the losses attained by the network during the training process, various different optimizers mentioned above. SGD performed best, while Adam was a close second. The deep learning community frequently claim that SGD works best on classification problems, and this may be the reason why SGD performed the best.

## Conclusion

The final model under 5 Million parameters and employing SGD achieved an accuracy of 95.55% on the CIFAR-10 test data set by systematically adjusting the hyperpameters and optimizers.

## Acknowledgement

We would like to thank everyone whose comments and suggestions helped us with the project. We appreciate the constant assistance of Professors Chinmay Hegde, Arsalan Mosenia, and the teaching assistant Teal Witter. Last but not least, we would like to express our sincere gratitude to the teaching staff for providing us with the chance to complete these tasks and projects. They were highly beneficial and relevant to comprehending the ideas.


