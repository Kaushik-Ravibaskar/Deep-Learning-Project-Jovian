# Birds Species Classifier (Small Project)

Hello all,
This is a small **Deep Learning** project which I did as a part of *Deep Learning with PyTorch: Zero to GANs* course offered by **Jovian**.

In this project, I have taken a dataset from **Kaggle** which involves images of birds *(of 224 X 224 resolution)* classified into 400 species. My model, which is based on the **ResNet9 architechture**, tries to classify the birds into their respective species.\
Link of the dataset: https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species

Here's a summary of the different techniques which I used in this model to improve the performance and reduce the training time:

**Data normalization**: Normalizing the image tensors by subtracting the mean and dividing by the standard deviation of pixels across each channel. Normalizing the data prevents the pixel values from any one channel from disproportionately affecting the losses and gradients.

**Data augmentation**: Applying random transformations while loading images from the training dataset. Specifically, we will pad each image by 20 pixels, and then take a random crop of size 64 x 64 pixels, and then flip the image horizontally with a 50% probability.

**Residual connections**: One of the key changes to our CNN model was the addition of the *residual block*, which adds the original input back to the output feature map obtained by passing the input through one or more convolutional layers.

**Batch normalization**: After each convolutional layer, I added a batch normalization layer, which normalizes the outputs of the previous layer. This is somewhat similar to data normalization, except it's applied to the outputs of a layer, and the mean and standard deviation are learned parameters.

**Learning rate scheduling**: Instead of using a fixed learning rate, I used a learning rate scheduler, which will change the learning rate after every batch of training. There are many strategies for varying the learning rate during training, and I used the **One Cycle Learning Rate Policy**.

**Weight Decay**: I added weight decay to the optimizer, yet another regularization technique which prevents the weights from becoming too large by adding an additional term to the loss function.

**Gradient clipping**: I also added gradient clipping, which helps limit the values of gradients to a small range to prevent undesirable changes in model parameters due to large gradient values during training.

**Adam optimizer**: Instead of SGD (stochastic gradient descent), I used the Adam optimizer which uses techniques like momentum and adaptive learning rates for faster training.

**Records**: https://jovian.ai/kaushikravibaskar/birds-species-classifier/v/18/records 

*Notebook used for reference: https://jovian.ai/aakashns/05b-cifar10-resnet* \
*A special thanks to Jovian for the wonderful course!*