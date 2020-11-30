# Assignment-9: Sentiment Analysis and Neural Embeddings

## TOC

1. [Overview](#overview)
2. [Sentiment Analysis](#sentiment-analysis)
3. [Model Deployment](#model-deployment)
4. [References](#references)

## Overview

This assignment is targeted to understand Neural Embeddings - the base layer from where the NLP journey starts.
The target is to train an NLP model to predict the sentiment of a movie review.

## Sentiment Analysis

Training code can be found in this [notebook](https://github.com/rajy4683/EVA4P2/blob/master/S9-SentimentAnalysis/EVA4P2_SentimentAnalysis.ipynb)

## Model Deployment

Given the size restrictions on Lambda the overall deployment is split into two Lambda applications:

1. A [Spacy Tokenizer](https://spacy.io/api/tokenizer) that can return an array of tokens given an English sentence.

    ```curl
        curl -X POST \
          https://n6axj3jn08.execute-api.ap-south-1.amazonaws.com/dev/classify \
          -H 'cache-control: no-cache' \
          -H 'content-type: application/json' \
          -H 'postman-token: fa591a88-e17e-013a-49e6-c2747cf2e25f' \
          -d '{"text":"Movie sucks"}'P
    ```

2. The main deployment model that invokes Spacy tokens and then finally runs the NLP model and responds with a sentiment score between 0(Negative) and 1(Positive)  

    ```curl
        curl -X POST \
          https://n6axj3jn08.execute-api.ap-south-1.amazonaws.com/dev/classify \
          -H 'cache-control: no-cache' \
          -H 'content-type: application/json' \
          -H 'postman-token: 36448107-0684-54d6-5928-4cf6ced2cbf9' \
          -d '{"text":"Movie sucks"}'
    ```

Spacy model is deployed on AWS Lambda and can be accessed at the below URL:

https://rekogwebservice.tk/

Widget Name:
Movie Review Sentiment Analyzer

Please note that the card widgets on the above page interact with AWS LAMBDA backend and suffer from a cold start issue.
Hence please retry clicking on the "Upload"/"Submit" Buttons

## References

- EVA4 Course content
- [Jinhyeok Yang's R1 GAN models](https://github.com/Yangyangii/GAN-Tutorial/blob/master/CelebA/R1GAN.ipynb)
