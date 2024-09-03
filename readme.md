Wine Quality Prediction
# Predicting Wine Quality with Neural Networks

This project implements a basic feedforward neural network from scratch to predict wine quality based on various chemical properties. The model is built without using libraries like NumPy or Pandas, making it a great learning tool for understanding the fundamentals of neural networks.

## Overview
Wine quality is influenced by several factors such as acidity, sugar content, and pH. This project aims to predict the quality of wine using a neural network with a single hidden layer. The network is trained on a dataset of wine characteristics and outputs a quality score.

## Features
- Feedforward Neural Network: The model includes an input layer, one hidden layer, and an output layer.
- Activation Functions: ReLU is used in the hidden layer, while a linear activation function is used in the output layer.
- Backpropagation: The network is trained using backpropagation to minimize the error in predictions.
- Manual Implementation: All neural network components (weights, biases, activation functions, etc.) are manually implemented without external libraries.

## Dataset
The model is trained on the Wine Quality Dataset, which contains chemical properties of red and white wine samples along with their quality ratings. The dataset can be downloaded for free from the UCI Machine Learning Repository:

- [Red Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

## Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/wine-quality-prediction.git
cd wine-quality-prediction
```

## Usage
### Prepare the Dataset:
1. Download the wine quality dataset from the link above.
2. Place the CSV file in the project directory.

### Train the Model:
Run the training script to train the neural network on the wine quality dataset:

```bash
python train.py
```

### Predict Wine Quality:
After training, you can use the model to predict the quality of new wine samples by running the prediction script:

```bash
python predict.py
```

## Project Structure
- `wine_quality_nn.py`: Contains the SimpleNN class, which defines the neural network.
- `train.py`: Script to train the neural network on the wine dataset.
- `predict.py`: Script to make predictions using the trained model.
- `README.md`: Project description and instructions.

## How It Works
1. Initialization: The neural network is initialized with random weights and biases.
2. Forward Pass: The input data passes through the network, and predictions are generated.
3. Backward Pass: The error between predictions and actual wine quality scores is calculated and used to adjust the weights and biases to improve accuracy.
4. Training: The network goes through multiple epochs of forward and backward passes, learning from the data.
5. Prediction: Once trained, the network can predict the quality of unseen wine samples.

## Limitations
- The model is a simple implementation and may not achieve high accuracy on complex datasets.
- Performance could be improved by using more layers, different activation functions, or optimizing hyperparameters.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code.

