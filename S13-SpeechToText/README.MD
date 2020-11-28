# Assignment-12: Speech to Text

## TOC

1. [Overview](#overview)
2. [Speech Recognition Pipeline](#speech-recognition-pipeline)
3. [Model Deployment](#model-deployment)
4. [References and Credits](#references-and-credits)

## Overview

In this assignment, we create an End-To-End Speech to text Pipeline.
The core model is based on Deep Speech2 architecture consisting of Residual CNN layers followed by BiDirectional RNN layers.
The ResNet layers are used for extracting audio features, which the RNN layers are responsible for predicting the temporal relation and making prediction for each frame of audio based on the data available till that point in time.
The code and inputs are inspired from this [blog](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch)

## Speech Recognition Pipeline

For training, a subset of [LibriSpeech dataset](http://www.openslr.org/12/) dataset has been used. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.
The link to the original paper can be found [here](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf)


The model consists of following elements:

1. ResNet Layers: We use 3 Layer Residual blocks for feature extraction. Also GELU activation replaces the standard RELU functions.
2. RNN Layers: We use Bi-Directional GRUs with 5 layers and 512 hidden_layer dimension.
3. CTC Loss :  Connection Temporal Loss function as implemented in PyTorch is used as loss criteria. Further we also evaluate the model based on standard WER (word error rate) and CER(character error rate)

Following transforms are applied before training the data:
1. The audio samples are converted to MelSpectograms with sampling rate of 16000 and max 128 mel samples are used for training
2. FrequencyMasking and TimeMasking is applied to block specific frequency dimensions and time slots respectively

Training code can be found in this [notebook](https://github.com/rajy4683/EVA4P2/blob/master/S13-SpeechToText/EVA4P2S13_EndToEndSpeech.ipynb)

## Model Deployment

The trained model is deployed on AWS Lambda and can be accessed as below:

    ```curl
		curl -X POST \
  				https://0bmpqw0kb9.execute-api.ap-south-1.amazonaws.com/dev/classify \
  				-H 'cache-control: no-cache' \
  				-H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  				-H 'postman-token: 862f22ee-dd51-c0e1-3143-9da4495a63b9' \
  				-F 'file=@audio_snip.mp3'
    ```
or on this page:

https://rekogwebservice2.s3-website.ap-south-1.amazonaws.com/ 

Card Widget: "Speech to Text"

The model currently accepts only MP3 encoded files and still in the process of refinement. Due to possible issues with sampling rate mismatch the model performance on real data is not great.
However, we will be enhancing this model shortly.

Please note that the card widgets on the above page interacts with AWS LAMBDA backend and hence suffers from a cold start issue.
So please retry by clicking on the "Upload"/"Submit" Buttons

## References and Credits

- EVA4 Course content
- [Assembly AI Blog](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch)
- [Sequence Modeling with CTC](https://distill.pub/2017/ctc/)
- [Speech Processing for ML](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
