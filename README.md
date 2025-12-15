# Neural Network Library & Advanced Applications  
CSE473 – Computational Intelligence, Fall 2025  
Faculty of Engineering, Ain Shams University

## Project Overview
This repository contains a complete neural network library implemented from scratch using Python and NumPy. The project demonstrates the full workflow of building a modular deep learning framework, validating it on the XOR problem, developing an autoencoder for MNIST reconstruction, and using the encoder as a feature extractor for SVM classification. A TensorFlow/Keras baseline is also included for comparison.

The project consists of four major components:
1. A custom neural network library.
2. XOR problem training and validation.
3. Autoencoder training and reconstruction analysis.
4. Latent‑space SVM classification and TensorFlow comparison.

## Features

### Custom Neural Network Library
Implemented entirely with NumPy:
- Base `Layer` abstraction with `forward` and `backward` methods.
- Dense (Fully Connected) layers with weight and bias gradients.
- Activation layers: ReLU, Sigmoid, Tanh, Softmax.
- Loss functions: Mean Squared Error (MSE).
- SGD optimizer with optional momentum.
- Sequential `Model` class handling forward pass, backward pass, and parameter updates.
- Gradient checking utilities for validating backpropagation.

### XOR Problem
- Architecture: 2–4–1 MLP with Tanh and Sigmoid activations.
- Loss: MSE.
- Optimizer: SGD.
- Demonstrates correct learning of the XOR mapping.

### Autoencoder for MNIST
- Encoder compresses 784‑dimensional input into a 32–64 dimensional latent space.
- Decoder reconstructs the original image.
- Trained using MSE loss in an unsupervised manner.
- Includes reconstruction visualizations and loss curves.

### Latent Space SVM Classification
- Encoder outputs used as feature vectors.
- SVM trained on MNIST labels.
- Evaluation includes accuracy, confusion matrix, and classification metrics.

### TensorFlow/Keras Baseline
A baseline implementation using TensorFlow/Keras is provided to compare:
- Training time
- Reconstruction loss
- Implementation complexity

## Repository Structure
```
├── .gitignore
├── README.md
├── requirements.txt
│
├── lib/
│   ├── __init__.py
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   ├── optimizer.py
│   ├── network.py
│   └── utils.py
│
├── notebooks/
│   └── XOR.ipynb
│   └── tensorflow_baseline.ipynb
│   └── autoencoder_mnist_classificationXOR.ipynb
│   └── load_model.ipynb
│
└── report/
    └── CSE473_Project_2025-Team2.pdf
```


## Part 1: Library Implementation and XOR Validation

### XOR Architecture
| Layer | Description | Shape |
|-------|-------------|--------|
| Input | XOR dataset | (4, 2) |
| Dense(4) | Hidden layer | (4, 4) |
| Tanh | Nonlinearity | (4, 4) |
| Dense(1) | Output layer | (4, 1) |
| Sigmoid | Final activation | (4, 1) |

### Training Configuration
- Optimizer: SGD (learning rate = 0.5)
- Loss: MSE
- Epochs: 350
- Batch size: 4

The model converges to the correct XOR mapping.

## Part 2: Autoencoder and Latent Space Classification

### Autoencoder
- Encoder: Dense + ReLU layers.
- Latent space: 32–64 dimensions.
- Decoder: Dense + ReLU/Sigmoid layers.
- Loss: MSE.
- Outputs include reconstruction loss curves and image comparisons.

### SVM Classification
- Latent vectors extracted from the encoder.
- SVM trained on MNIST labels.
- Evaluation includes accuracy, confusion matrix, and classification metrics.

## TensorFlow/Keras Comparison
A baseline implementation using TensorFlow/Keras is provided to compare:
- Training time
- Reconstruction loss
- Implementation complexity

## Gradient Checking
The notebook includes numerical gradient checking using:




$ \frac{\partial L}{\partial W} \approx \frac{L(W + \epsilon) - L(W - \epsilon)}{2\epsilon} $




This validates the correctness of the analytical gradients.

## Installation
```
git clone <your-repo-url>
cd Neural-Network-Library-Advanced-Applications
pip install -r requirements.txt
```




The notebook includes:
1. Gradient checking  
2. XOR training  
3. Autoencoder training  
4. Latent space SVM classification  
5. TensorFlow comparison  

## Report
A detailed PDF report is included in:
``` report\CSE473_Project_2025-Team2.pdf ```


It covers:
- Library design and architecture
- XOR results
- Autoencoder reconstruction analysis
- SVM classification performance
- TensorFlow comparison
- Challenges and lessons learned

## Team Members
- Omar Khaled Ahmed  
- Marwan Mahmoud Ali  
- Mohamed Alaa Abdelkareem  
- Omar Emad El Din Hassan  
- Adham Waleed Gamal  

## License
This project is for academic use under the CSE473 course guidelines.

