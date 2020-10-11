# Assignment-8: Super Resolution and Neural Style Transfer

## TOC

1. [Overview](#overview)
2. [SRGAN Model for SuperResolution](#srgan-model-for-superresolution)
3. [Samples for SuperRes](#samples-for-superres)
4. [Neural Style Transfer](#neural-style-transfer)
5. [Samples for Neural Style Transfer](#samples-for-neural-style-transfer)
7. [Model Deployment](#model-deployment)
8. [References](#references)

## Overview

This assignment consists of two parts

1. Transfer Learning on a pretrained SRGAN Model to create Super Resolution Images for custom Dataset (Drones/Flying Birds/QuadCopters)

2. Training and Deployment of Neural Style Transfer on custom style images.

## SRGAN Model for SuperResolution

Training code can be found in this [notebook](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/EVA4P2_SuperRes_SRGANs.ipynb)

## Samples for SuperRes

![Samples 1](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/superressamples/sample1.JPG)
![Samples 2](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/superressamples/sample2.JPG)
![Samples 3](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/superressamples/sample3.JPG)

## Neural Style Transfer

16 different style images were used to train on COCO Val-2017 dataset.  

Training code can be found in this [notebook](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/EVA4P2_NeuralStyleTransfer.ipynb)

The models are deployed on AWS and can be accessed from the below [pages](#model-deployment).

## Samples for Neural Style Transfer

![Samples 1](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/nstsamples/sample1.JPG)
![Samples 2](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/nstsamples/sample2.JPG)
![Samples 3](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/nstsamples/sample3.JPG)
![Samples 4](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/nstsamples/sample4.JPG)
![Samples 5](https://github.com/rajy4683/EVA4P2/blob/master/S8-SuperResNST/nstsamples/sample5.JPG)

## Model Deployment

Super Resolution model is deployed on AWS Lambda and can be accessed at the below URL:

http://rekogwebservice2.s3-website.ap-south-1.amazonaws.com/superres

Neural Style Transfer link is deployed below:

http://rekogwebservice2.s3-website.ap-south-1.amazonaws.com/styletransfer

Please note that the card widgets on the above page interact with AWS LAMBDA backend and suffer from a cold start issue.
Hence please retry clicking on the "Upload"/"Submit" Buttons

## References

- EVA4 Course content
- [Jinhyeok Yang's R1 GAN models](https://github.com/Yangyangii/GAN-Tutorial/blob/master/CelebA/R1GAN.ipynb)
