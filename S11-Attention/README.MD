# Assignment-11: Attention mechanism and NMT

## TOC

1. [Overview](#overview)
2. [Seq2Seq Translation with Bahdnau Attention](#seq2seq-translation-with-bahndau-attention)
    1. [TorchScript Handling](#torchscript-handling)
1. [Model Deployment](#model-deployment)
1. [References](#references)

## Overview

This assignment is targeted to understand and implement Attention mechanism in Encoder/Decoder networks.
The original paper on [Bahdanau attention](https://arxiv.org/pdf/1409.0473.pdf).
The original notebook which forms the basis for this notebook can be found [here](https://bastings.github.io/annotated_encoder_decoder/)

## Seq2Seq Translation with Bahdnau Attention

The model consists of following elements:

1. Encoder Class: Implemented as a bi-directional GRU
1. Decoder Class: Implemented as a conditional GRU, where the initial hidden state of the decoder is conditioned on the encoder's final vector
1. Attention Mechanism: An additive attention is used at every time-step of the decoder.  
1. Embeddings: PyTorch's nn.Embedding class is used to derive embeddings from the vocabulary.

PyTorch's IWSLT German-English Translation dataset has been used to train the model.
Adam optimizer with default settings (β1=0.9, β2=0.999 and ϵ=10−8). Cross-Entropy loss is used as the cost-function.

Training code can be found in this [notebook](https://github.com/rajy4683/EVA4P2/blob/master/S11-Attention/EVA4P2S11_Attention.ipynb)

### TorchScript handling

TorchScript doesn't allow to invokde sub-modules once top level module has been traced. 
The model used in this assignment has a top-level EncoderDecoder class encompassing the Encoder and Decoder objects.
To handle this situation during tracing all the classes are separtely traced i.e Encoder, Decoder, Generator(Final FC Layer) and the Embedding Layers

## Model Deployment

The main deployment model that invokes Spacy tokens and then finally runs the NLP model and responds with a sentiment score between 0(Negative) and 1(Positive)  

    ```curl
        curl -X POST \
          https://qiyyo3c4xa.execute-api.ap-south-1.amazonaws.com/dev/classify \
          -H 'cache-control: no-cache' \
          -H 'content-type: application/json' \
          -H 'postman-token: 36448107-0684-54d6-5928-4cf6ced2cbf9' \
          -d '{"text":"Neuronale maschinelle Übersetzung macht absoluten Spaß"}'
    ```

https://rekogwebservice.tk/

Card Widget: "German to English Translator"

Please note that the card widgets on the above page interacts with AWS LAMBDA backend and hence suffers from a cold start issue.
So please retry by clicking on the "Upload"/"Submit" Buttons

## References

- EVA4 Course content
- [The Annotated Encoder Decoder](https://bastings.github.io/annotated_encoder_decoder/)
- [DEPLOYING A SEQ2SEQ MODEL WITH TORCHSCRIPT](https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html)
