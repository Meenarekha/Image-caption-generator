# Image Caption Generator using CNN-LSTM with Attention

An image captioning system that generates natural-language descriptions for input images using deep learning. The system combines **ResNet-50** for image feature extraction, **LSTM** for sequence generation, and an **Attention mechanism** to focus on relevant visual features while generating captions.

## Overview

The project follows an encoder-decoder architecture:

**Input Image → ResNet-50 → Image Features → Attention → LSTM → Generated Caption**

The trained model is integrated into a **Streamlit web application** for interactive image caption generation.

## Key Features

- Image feature extraction using **ResNet-50**
- Caption generation using **LSTM**
- Attention mechanism for relevant feature selection
- Vocabulary-based sequence generation
- Interactive Streamlit interface
- Pre-trained model inference

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **ResNet-50**
- **LSTM**
- **Attention Mechanism**
- **Streamlit**
- **Pickle**

## Dataset

**Flickr8k Dataset**

The project uses the Flickr8k image-caption dataset for training the image captioning model.

## Project Structure

```text
Image-caption-generator/
│
├── app.py
├── attention_model.pth
├── vocab_attention.pkl
├── requirements.txt
├── README.md
└── .gitignore
