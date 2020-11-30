# Assignment-6: Generative Adversarial Networks

## TOC

1. [Overview](#overview)
2. [R1 GAN Model Summary](#r1-gan-model-summary)
    1. [Generator](#generator)
    1. [Discriminator](#discriminator)
1. [Loss Graphs](#loss-graphs)
1. [Samples](#samples)
1. [References](#references)

## Overview

In this assignment, we train a GAN to generate cars available in the Indian sub-continent. The training set consists of approximately 2K images with either white or black backgrounds.
R1 GAN has been trained on this custom dataset:

1. All images are re-scaled and center-cropped to 128x128 for usage with Discriminator network
2. Generator network uses latent space with dimension of 256.
3. The Generator network outputs a 128x128 images
4. Discriminator uses [Softplus loss](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html) (basically a smoothed Relu Loss) with a Gradient Penalty. 

## R1 GAN Model Summary

### Generator

           ----------------------------------------------------------------
                    Layer (type)               Output Shape         Param #
            ================================================================
                        Linear-1             [-1, 1, 65536]      16,842,752
                        Conv2d-2            [-1, 512, 8, 8]       4,719,104
                          ReLU-3            [-1, 512, 8, 8]               0
                        Conv2d-4            [-1, 512, 8, 8]       2,359,808
                          ReLU-5            [-1, 512, 8, 8]               0
                        Conv2d-6            [-1, 512, 8, 8]         524,800
            R1GANResidualBlock-7            [-1, 512, 8, 8]               0
                      Upsample-8          [-1, 512, 16, 16]               0
                        Conv2d-9          [-1, 256, 16, 16]       1,179,904
                         ReLU-10          [-1, 256, 16, 16]               0
                       Conv2d-11          [-1, 256, 16, 16]         590,080
                         ReLU-12          [-1, 256, 16, 16]               0
                       Conv2d-13          [-1, 256, 16, 16]         131,328
            R1GANResidualBlock-14          [-1, 256, 16, 16]               0
                     Upsample-15          [-1, 256, 32, 32]               0
                       Conv2d-16          [-1, 128, 32, 32]         295,040
                         ReLU-17          [-1, 128, 32, 32]               0
                       Conv2d-18          [-1, 128, 32, 32]         147,584
                         ReLU-19          [-1, 128, 32, 32]               0
                       Conv2d-20          [-1, 128, 32, 32]          32,896
            R1GANResidualBlock-21          [-1, 128, 32, 32]               0
                     Upsample-22          [-1, 128, 64, 64]               0
                       Conv2d-23           [-1, 64, 64, 64]          73,792
                         ReLU-24           [-1, 64, 64, 64]               0
                       Conv2d-25           [-1, 64, 64, 64]          36,928
                         ReLU-26           [-1, 64, 64, 64]               0
                       Conv2d-27           [-1, 64, 64, 64]           8,256
            R1GANResidualBlock-28           [-1, 64, 64, 64]               0
                     Upsample-29         [-1, 64, 128, 128]               0
                       Conv2d-30         [-1, 64, 128, 128]          36,928
                         ReLU-31         [-1, 64, 128, 128]               0
                       Conv2d-32         [-1, 64, 128, 128]          36,928
                         ReLU-33         [-1, 64, 128, 128]               0
            R1GANResidualBlock-34         [-1, 64, 128, 128]               0
                       Conv2d-35          [-1, 3, 128, 128]           1,731
            ================================================================
            Total params: 27,017,859
            Trainable params: 27,017,859
            Non-trainable params: 0
            ----------------------------------------------------------------
            Input size (MB): 0.00
            Forward/backward pass size (MB): 78.38
            Params size (MB): 103.06
            Estimated Total Size (MB): 181.44

### Discriminator

        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv2d-1         [-1, 64, 128, 128]           1,792
                    Conv2d-2        [-1, 128, 128, 128]          73,856
                      ReLU-3        [-1, 128, 128, 128]               0
                    Conv2d-4        [-1, 128, 128, 128]         147,584
                      ReLU-5        [-1, 128, 128, 128]               0
                    Conv2d-6        [-1, 128, 128, 128]           8,320
        R1GANResidualBlock-7        [-1, 128, 128, 128]               0
                 AvgPool2d-8          [-1, 128, 64, 64]               0
                    Conv2d-9          [-1, 256, 64, 64]         295,168
                     ReLU-10          [-1, 256, 64, 64]               0
                   Conv2d-11          [-1, 256, 64, 64]         590,080
                     ReLU-12          [-1, 256, 64, 64]               0
                   Conv2d-13          [-1, 256, 64, 64]          33,024
        R1GANResidualBlock-14          [-1, 256, 64, 64]               0
                AvgPool2d-15          [-1, 256, 32, 32]               0
                   Conv2d-16          [-1, 512, 32, 32]       1,180,160
                     ReLU-17          [-1, 512, 32, 32]               0
                   Conv2d-18          [-1, 512, 32, 32]       2,359,808
                     ReLU-19          [-1, 512, 32, 32]               0
                   Conv2d-20          [-1, 512, 32, 32]         131,584
        R1GANResidualBlock-21          [-1, 512, 32, 32]               0
                AvgPool2d-22          [-1, 512, 16, 16]               0
                   Conv2d-23         [-1, 1024, 16, 16]       4,719,616
                     ReLU-24         [-1, 1024, 16, 16]               0
                   Conv2d-25         [-1, 1024, 16, 16]       9,438,208
                     ReLU-26         [-1, 1024, 16, 16]               0
                   Conv2d-27         [-1, 1024, 16, 16]         525,312
        R1GANResidualBlock-28         [-1, 1024, 16, 16]               0
                AvgPool2d-29           [-1, 1024, 8, 8]               0
                   Linear-30                    [-1, 1]          65,537
        ================================================================
        Total params: 19,570,049
        Trainable params: 19,570,049
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.19
        Forward/backward pass size (MB): 195.50
        Params size (MB): 74.65
        Estimated Total Size (MB): 270.34
        ----------------------------------------------------------------    

## Loss Graphs

![Loss Graphs](https://github.com/rajy4683/EVA4P2/blob/master/S6-GANs/losses.png)

## Samples

![Samples](https://github.com/rajy4683/EVA4P2/blob/master/S6-GANs/samples.png)

## Latent Space Exploration

![Samples](https://github.com/rajy4683/EVA4P2/blob/master/S6-GANs/modeltransitions.png)

## Model Deployment

This model is deployed on AWS Lambda and can be accessed at the below URL(Cars!Real or Fake? card):

https://rekogwebservice.tk/

Card Widget:
"Cars! Real or Fake?"

Please note that the card widgets on the above page interact with AWS LAMBDA backend and suffer from a cold start issue.
Hence please retry clicking on the "Upload"/"Submit" Buttons

## References

- EVA4 Course content
- [Jinhyeok Yang's R1 GAN models](https://github.com/Yangyangii/GAN-Tutorial/blob/master/CelebA/R1GAN.ipynb)
