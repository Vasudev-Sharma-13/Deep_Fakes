# Deep_Fakes
Deep Fakes using Keras/tensorflow and MNIST dataset(Coursera supervision)

# MNIST GAN: Generative Adversarial Network for Digit Generation


This repository contains a Python script that implements a Generative Adversarial Network (GAN) for generating images of the digit '0' from the MNIST dataset. GANs consist of two neural networks, a generator, and a discriminator, that are trained simultaneously through a competitive process. The generator aims to produce realistic images, while the discriminator aims to distinguish between real and generated images. This competition drives both networks to improve their performance over time.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/index.html)

You can install these libraries using pip:

'''bash'''
pip install tensorflow numpy matplotlib pillow
## Code Structure

**1. Data Loading and Preprocessing**

The code loads the MNIST dataset, extracts images of the digit '0,' and normalizes the pixel values to the range [0, 1].

**2. Discriminator**

The discriminator network is responsible for distinguishing between real and generated images. It consists of convolutional layers followed by batch normalization and a final sigmoid activation function.

**3. Generator**

The generator network aims to create realistic images from random noise. It consists of dense layers followed by convolutional transpose layers, batch normalization, and a final sigmoid activation function.

**4. GAN (Generative Adversarial Network)**

The GAN combines the generator and discriminator into a single model. During training, the generator tries to produce images that the discriminator cannot distinguish from real ones.

**5. Training Loop**

The code trains the GAN through a series of epochs. In each epoch, it iterates through the dataset, training the discriminator and generator alternately. At the end of each epoch, it generates and displays a sample image to visualize the progress.



