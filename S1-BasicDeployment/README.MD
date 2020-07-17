# Assignment-1: Deploying a model on AWS Lambda

## Description

 This repo contains the source code for deploying PyTorch based pre-trained [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)  on AWS Lambda.
 The deployed model can be used for inferencing of images belonging to the [Imagenet classes](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
 
## File-wise Details

- [handler.py](https://github.com/rajy4683/EVA4P2/blob/master/S1-BasicDeployment/handler.py): Contains the definition of the lambda function(classify_image) that will invoked on the AWS LAMBDA
- [serverless.yml](https://github.com/rajy4683/EVA4P2/blob/master/S1-BasicDeployment/serverless.yml): Deployment file used by the "serverless" framework to create the end-to-end deployment.
- [package.json](https://github.com/rajy4683/EVA4P2/blob/master/S1-BasicDeployment/package.json): contains the dependency file used during 'npm run deploy' phase.
- [requirements.txt](https://github.com/rajy4683/EVA4P2/blob/master/S1-BasicDeployment/requirements.txt): contains all the external packages that need to be deployed for 'handler.py'
- [imagenet_labels.py](https://github.com/rajy4683/EVA4P2/blob/master/S1-BasicDeployment/imagenet_labels.py): contains an Unordered dict of Imagenet class to name mappings

## Samples from Inferencing

![Input Image for the query](https://github.com/rajy4683/EVA4P2/blob/master/S1-BasicDeployment/LABD.jpg)

![Output of Inferencing from deployed model using Insomnia](https://github.com/rajy4683/EVA4P2/blob/master/S1-BasicDeployment/Output.JPG)
