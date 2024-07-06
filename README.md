# Transformers_from_scratch
This is the implementation from umar jamil youtube video on transformers : https://youtu.be/ISNdQcPhsts?si=lZu4eC-X1bBVxwjI



# Transformer Model in PyTorch

This repository contains the implementation of a Transformer model in PyTorch, including scripts for training the model on bilingual datasets.

## Table of Contents
- [Introduction](#introduction)
- [Transformer Diagram](#transformer-diagram)
- [Components](#components)
- [Usage](#usage)
- [Training](#training)
- [License](#license)

## Introduction

This implementation provides a modular and extensible structure for the Transformer model. Each component of the model is implemented as a separate class, making it easy to understand and modify individual parts of the architecture.

## Transformer Diagram

The Transformer model architecture is a complex structure best understood visually. Below is the diagram to help illustrate the concept:

### Technical Diagram

For a detailed technical diagram of the Transformer architecture, please refer to this image:

![Transformer Architecture](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*BHzGVskWGS_3jEcYYi6miQ.png)

This diagram shows the intricate components of the Transformer model, including the encoder and decoder stacks, multi-head attention mechanisms, and feed-forward networks.

### Pop Culture Reference

For a lighter take on "transformers", here's an image from the popular movie franchise:

![Transformers Movie](https://www.comingsoon.net/wp-content/uploads/sites/3/2023/06/Watch-the-Transformers-Movies-Before-Rise-of-the-Beasts.jpg?resize=1024,576)

While not technically related to our NLP model, this image serves as a fun reminder of the term "transformer" in popular culture!

Our implementation focuses on the technical architecture shown in the first image, not the movie characters. The model transforms input sequences into output sequences, hence the name "Transformer".
## Components

### model.py

Contains the implementation of various components of the Transformer model such as:
- LayerNormalization
- FeedForwardBlock
- InputEmbeddings
- PositionalEncoding
- ResidualConnection
- MultiHeadAttentionBlock
- EncoderBlock
- Encoder
- DecoderBlock
- Decoder
- ProjectionLayer
- Transformer

### dataset.py

Contains the implementation of:
- BilingualDataset: Handles the preparation of bilingual data for the Transformer model, including tokenization, padding, and masking.
- causal_mask: Creates a causal mask for the decoder to ensure that the model does not attend to future positions in the sequence.

### train.py

Handles the training of the Transformer model. Includes functions such as:
- `greedy_decode`: Performs greedy decoding to generate translations from the model.
- `run_validation`: Runs the validation process and calculates evaluation metrics.
- `get_all_sentences`: Yields all sentences in a dataset for a given language.
- `get_or_build_tokenizer`: Builds or loads a tokenizer.
- `get_ds`: Loads the dataset, builds tokenizers, and creates dataloaders.
- `get_model`: Builds the Transformer model.
- `train_model`: Orchestrates the training process.

## Usage

To build and use the Transformer model, you need to follow these steps:

1. **Prepare the dataset**: Ensure you have a bilingual dataset to train on.
2. **Configure the model**: Set up the configuration parameters for the model and training process.
3. **Run the training script**: Execute `train.py` to start the training process.

## Training

To train the model, execute the `train.py` script:

```bash
python train.py
```


