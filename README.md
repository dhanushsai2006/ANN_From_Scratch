# Neural Networks from Scratch

A comprehensive implementation of neural networks built from the ground up using only NumPy, providing deep insights into the mathematical foundations of deep learning.

## Overview

This project demonstrates how to build neural networks without relying on high-level deep learning frameworks like TensorFlow or PyTorch. By implementing everything from scratch, you'll gain a thorough understanding of:

- Forward propagation algorithms
- Backpropagation and gradient computation
- Weight initialization strategies
- Activation functions and their derivatives
- Loss functions and optimization techniques

## Features

- **Pure NumPy Implementation**: No external deep learning libraries required
- **Modular Architecture**: Clean, well-structured code that's easy to understand and modify
- **Multiple Activation Functions**: Support for ReLU, Sigmoid, Tanh, and more
- **Flexible Network Architecture**: Easily configurable layers and neurons
- **Comprehensive Documentation**: Detailed explanations of mathematical concepts
- **Visual Learning**: Includes diagrams and visualizations of key concepts

## Project Structure

```
├── model.py              # Core neural network implementation
├── NN-from-Scratch.ipynb # Interactive tutorial notebook
├── figs/                 # Visualization and diagram assets
│   └── backprop_algo_backward.png
├── examples/             # Usage examples and demonstrations
└── README.md            # Project documentation
```

## Quick Start

### Requirements

```bash
pip install numpy matplotlib
```

### Basic Usage

```python
import numpy as np
from model import NeuralNetwork

# Create a simple neural network
nn = NeuralNetwork(layers=[784, 128, 64, 10])

# Train on your data
nn.train(X_train, y_train, epochs=100, learning_rate=0.01)

# Make predictions
predictions = nn.predict(X_test)
```

## Mathematical Foundation

### Forward Propagation

The forward pass computes the output of each layer:

```
z^(l) = W^(l) * a^(l-1) + b^(l)
a^(l) = σ(z^(l))
```

Where:
- `z^(l)` is the linear combination at layer l
- `W^(l)` are the weights
- `b^(l)` are the biases
- `σ` is the activation function

### Backpropagation

The backward pass computes gradients using the chain rule:

```
∂L/∂W^(l) = ∂L/∂z^(l) * ∂z^(l)/∂W^(l)
∂L/∂b^(l) = ∂L/∂z^(l)
```

## Key Components

### Activation Functions
- **ReLU**: `f(x) = max(0, x)`
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- **Tanh**: `f(x) = tanh(x)`
- **Softmax**: For multi-class classification

### Loss Functions
- **Mean Squared Error**: For regression tasks
- **Cross-Entropy**: For classification tasks
- **Binary Cross-Entropy**: For binary classification

### Optimization
- **Gradient Descent**: Basic optimization algorithm
- **Momentum**: Accelerated gradient descent
- **Learning Rate Scheduling**: Adaptive learning rates

## Examples

### Binary Classification

```python
# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])
nn.train(X, y, epochs=1000, learning_rate=0.1)
```

### Multi-class Classification

```python
# MNIST-like dataset
nn = NeuralNetwork([784, 128, 64, 10])
nn.train(X_train, y_train, epochs=50, learning_rate=0.01)

accuracy = nn.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}%")
```

## Learning Resources

### Understanding Backpropagation
The backpropagation algorithm is the heart of neural network training. It efficiently computes gradients by propagating errors backward through the network, allowing us to update weights to minimize the loss function.

### Weight Initialization
Proper weight initialization is crucial for successful training. This implementation includes several initialization strategies:
- Xavier/Glorot initialization
- He initialization
- Random normal initialization

### Regularization Techniques
- L1/L2 regularization to prevent overfitting
- Dropout simulation
- Early stopping criteria

## Performance Tips

1. **Batch Processing**: Process multiple samples simultaneously for efficiency
2. **Vectorization**: Leverage NumPy's vectorized operations
3. **Learning Rate Tuning**: Start with 0.01 and adjust based on convergence
4. **Feature Scaling**: Normalize inputs for faster convergence

## Troubleshooting

### Common Issues
- **Vanishing Gradients**: Use ReLU activation or gradient clipping
- **Exploding Gradients**: Reduce learning rate or use gradient clipping
- **Slow Convergence**: Check learning rate and initialization
- **Overfitting**: Add regularization or reduce model complexity

## Contributing

Feel free to contribute by:
- Adding new activation functions
- Implementing optimization algorithms
- Improving documentation
- Adding more examples
