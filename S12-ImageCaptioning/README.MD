# Assignment-12: Image Captioning

## TOC

1. [Overview](#overview)
2. [Image Captioning with SANT](#image-captioning-with-sant)
3. [Model Deployment](#model-deployment)
4. [References](#references)

## Overview

This assignment is targeted to understand and implement Show, Attend and Tell Paper for Image captioning
The original paper can be found here **[Show, Attend, and Tell](https://arxiv.org/abs/1502.03044)**
Please refer to [this repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) for complete details and explanations.

## Image Captioning with SANT

The model consists of following elements:

1. Encoder Class: Is a standard ResNet-101 model from which the last layer has been removed. The terminal layer produces an encoding of the same [Batch_size, 14, 14, 2048]
2. Decoder Class: Implemented as a uni-directional LSTM, where the initial hidden state of the decoder is conditioned on the encoder's output
3. Attention Mechanism: An additive attention is used at every time-step of the decoder. Weighted encoded images indicate at each time step indicate where the Decoder should pay attention.
4. Embeddings: PyTorch's nn.Embedding class is used to derive embeddings from the vocabulary.
5. Beam Search: Described in [this link](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#beam-search)


Training code can be found in this [notebook](https://github.com/rajy4683/EVA4P2/blob/master/S12-ImageCaptioning/EVA4P2S12_ImageCaptionFinal.ipynb)

## Model Deployment

The main deployment model that invokes Spacy tokens and then finally runs the NLP model and responds with a sentiment score between 0(Negative) and 1(Positive)  

    ```curl
		curl -X POST \
  				https://3kpbfogkt0.execute-api.ap-south-1.amazonaws.com/dev/classify \
  				-H 'cache-control: no-cache' \
  				-H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  				-H 'postman-token: 862f22ee-dd51-c0e1-3143-9da4495a63b9' \
  				-F 'file=@flickr30k_images/3826378639.jpg'
    ```

http://rekogwebservice2.s3-website.ap-south-1.amazonaws.com/ 

Card Widget: "Caption This!"

Please note that the card widgets on the above page interacts with AWS LAMBDA backend and hence suffers from a cold start issue.
So please retry by clicking on the "Upload"/"Submit" Buttons

## References and Credits

- EVA4 Course content
- [The Annotated Encoder Decoder](https://bastings.github.io/annotated_encoder_decoder/)
- [PyTorch tutorial for image captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- [DEPLOYING A SEQ2SEQ MODEL WITH TORCHSCRIPT](https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html)
