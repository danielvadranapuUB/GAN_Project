# Final Project Instructions to run the Code



## Requirements

* Ubuntu 22.04, Nvidia Graphics Card atleast 40GB, 32GB cpu ram, Cuda 12.0  
* Python3.8 is required to run the LDM repository
* COCO Dataset

## Weights Download link

[https://drive.google.com/drive/folders/1LZYRykzSzItOhJKM_TqZmcfbYdAvOVbt?usp=sharing](https://drive.google.com/drive/folders/1LZYRykzSzItOhJKM_TqZmcfbYdAvOVbt?usp=sharing)

## Dependencies 

```cmd
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
```

## Running LDM Script

* Load the code in colab and run Inferencing_LDM.ipynb cells 

## Results using LDM model

![](overleaf_files/a_happy_bear.png)

![](overleaf_files/a_happy_beer.png)



## Running Conditional GAN

# Conditional GAN for Text-to-Image Synthesis

This project implements a Conditional Generative Adversarial Network (GAN) to generate images based on textual descriptions. The model uses a pre-trained BERT model for text encoding and a deep convolutional neural network for the generator and discriminator.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Dataset

The model is trained on the COCO dataset, specifically the `train2017` subset along with its annotations. The COCO dataset contains images with corresponding textual descriptions (captions).

- **Image Directory:** The directory containing the training images.
- **Annotations File:** JSON file containing image captions.

## Model Architecture

### Text Encoder

- **BERT-Based Text Encoder:** Uses a pre-trained BERT model to convert text descriptions into feature vectors.

### Generator

- **Input:** Concatenates noise vector and text embedding.
- **Architecture:**
  - Fully Connected Layer
  - Three Transposed Convolutional (Deconvolutional) Layers
  - ReLU activations followed by Tanh activation

### Discriminator

- **Input:** Takes an image and a text embedding.
- **Architecture:**
  - Three Convolutional Layers
  - LeakyReLU activations
  - Fully Connected Layer
  - Sigmoid activation

## Running Text to Image GAN model

* Run cgan_bert.ipynb and cgan_bert_pnp.ipynb 

# Results

![](overleaf_files/test4_stitched_image.png)

![Model Loss](overleaf_files/test4_plots.png)

# Image to Image Generation on Animal Face Dataset

* run cgan_afhq.ipynb 

## Results

![](overleaf_files/cgan_loss.png)

![](overleaf_files/cat.png)