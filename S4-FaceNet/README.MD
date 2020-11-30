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

## Face Recognition Pipeline

Please refer this notebook for **[Face Recognition Pipeline](https://github.com/rajy4683/EVA4P2/blob/master/S4-FaceNet/P2S4_FaceNet.ipynb)**

Deployment Scripts are present in **[this repo](https://github.com/rajy4683/EVA4P2/tree/master/S4-FaceNet/s4-facedetection)**

### Loss Graph


## Deployment

Please visit the following link to view/test the deployed models:
https://rekogwebservice.tk/

## References

- EVA4 Course content
- [Finetune a facial recognition classifier](https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79)
- [Tim Esler's Awesome FaceNet Repo](https://github.com/timesler/facenet-pytorch)
