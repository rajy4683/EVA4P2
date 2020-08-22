# Assignment-4: Face Recognition Pipeline on custom images + LFW 

## TOC

1. [Overview](#overview)
2. [Face Recognition Pipeline](#face-recognition-pipeline)
    1. Loss Graph(#loss-graph)
1. [Deployment](#deployment)
1. [References](#references)

## Overview

In this assignment, the objective is to create an face recognition pipeline on custom dataset of personalities. Following are the stages:

1. Dataset creation and merge with LFW dataset
2. Extract and Align 160px images from the dataset and convert it for Face Recognition
3. Fine-tune InceptionResNetv1 pre-trained on VggFace2 for 

## Face Alignment

Please refer this notebook for **[Face Alignment](https://github.com/rajy4683/EVA4P2/blob/master/S3-FaceAlignSwap/P2S3_FaceAlignment.ipynb)**

Deployment Scripts are present in **[this repo](https://github.com/rajy4683/EVA4P2/tree/master/S3-FaceAlignSwap/s3-facealign)**

## Face Swap

Please refer this notebook for **[Face Swap](https://github.com/rajy4683/EVA4P2/blob/master/S3-FaceAlignSwap/P2S3_FaceSwap.ipynb)**

Deployment Scripts are present in **[this repo](https://github.com/rajy4683/EVA4P2/tree/master/S3-FaceAlignSwap/s3-faceswap)**

### Preview

![Sample Images](https://github.com/rajy4683/EVA4P2/blob/master/S3-FaceAlignSwap/SwapResults.png)

## Works Page

Please visit the following link to view/test the deployed models:
https://rekogwebservice2.s3-website.ap-south-1.amazonaws.com

## References

- EVA4 Course content
- [Finetune a facial recognition classifier](https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79)
- 
