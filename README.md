# 🚀 Neural Networks from Scratch

🧠 **Build Neural Networks Without Deep Learning Frameworks - Just NumPy!**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org)

## 📚 Overview

This project demonstrates how to build neural networks from scratch using only NumPy, without relying on high-level deep learning frameworks. Perfect for understanding the fundamental concepts behind neural networks! 


## ✨ Features

- 🏗️ **3-Layer Architecture**: Input → Hidden → Output layers
- 🎭 **Multiple Activation Functions**: ReLU and Sigmoid implementations
- 🚀 **Optimizers**: SGD and Momentum optimization
- 📈 **MNIST Dataset**: Train on handwritten digit recognition
- ⚡ **Fast Training**: ~98% accuracy in just 10 seconds on CPU
- 📓 **Jupyter Notebook**: Interactive learning experience

## 🏃‍♂️ Quick Start

### Prerequisites
```bash
pip install numpy matplotlib
```

### 🚀 Basic Training
```bash
python train.py
```

### 🎛️ Advanced Configuration
```bash
# Using sigmoid activation with momentum optimizer
python train.py --activation sigmoid --optimizer momentum --l_rate 4

# Custom batch size and learning rate
python train.py --batch_size 64 --l_rate 0.01 --beta 0.9
```

## ⚙️ Configuration Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--activation` | `relu`, `sigmoid` | 🎭 Activation function |
| `--optimizer` | `sgd`, `momentum` | 🏃‍♂️ Optimization algorithm |
| `--batch_size` | Integer | 📦 Training batch size |
| `--l_rate` | Float | 📈 Learning rate |
| `--beta` | Float | 🎯 Beta parameter for momentum |

## 🏗️ Architecture

### 🔢 Network Structure
- **Input Layer**: 784 nodes (28×28 flattened images)
- **Hidden Layer**: 64 nodes  
- **Output Layer**: 10 nodes (digit classes 0-9)

### 🧮 Mathematical Foundation

**Forward Propagation:**
```
Z₁ = W₁X + b₁
A₁ = activation(Z₁)
Z₂ = W₂A₁ + b₂  
A₂ = softmax(Z₂)
```

**Backward Propagation:**
```
∂L/∂W = (1/m) * A * X^T
∂L/∂b = (1/m) * Σ(∂L/∂Z)
```

## 📊 Dataset

🔢 **MNIST Handwritten Digits**
- 📚 70,000 total images
- 🏋️ 60,000 training samples
- 🧪 10,000 testing samples
- 📏 28×28 pixel grayscale images
- 🎯 10 classes (digits 0-9)

## 🚀 Performance

- ⚡ **Training Time**: ~10 seconds on CPU
- 🎯 **Accuracy**: ~98% on test set
- 💾 **Memory Efficient**: Pure NumPy implementation
- 🔧 **Lightweight**: No external ML frameworks required

## 📁 Project Structure

```
📦 neural-networks-scratch/
├── 🐍 train.py              # Main training script
├── 🧠 model.py              # Neural network implementation
├── 📓 NN-from-Scratch.ipynb # Interactive Jupyter notebook
├── 📊 data/                 # Dataset directory
├── 🖼️ figs/                 # Visualization figures
└── 📚 README.md             # This file
```

## 🎓 Educational Value

### 🔬 Core Concepts Covered
- 🧮 **Matrix Operations**: Understanding dot products and dimensions
- 📐 **Calculus**: Gradients and chain rule in backpropagation  
- 📊 **Statistics**: Weight initialization and normalization
- 🎯 **Optimization**: Gradient descent and momentum

### 🎭 Activation Functions
```python
# ReLU Activation
def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

# Sigmoid Activation  
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))
```

## 🎨 Visualization

The project includes beautiful visualizations of:
- 📈 Training loss curves
- 🎯 Accuracy progression  
- 🔍 Weight distributions
- 🧠 Network architecture diagrams

## 🔧 Advanced Usage

### 🎮 Interactive Mode
```python
from model import DeepNeuralNetwork

# Initialize network
dnn = DeepNeuralNetwork(sizes=[784, 64, 10], activation='relu')

# Train the model
dnn.train(x_train, y_train, x_test, y_test)

# Make predictions
predictions = dnn.predict(x_new)
```

### 📊 Custom Experiments
Try experimenting with:
- 🏗️ Different architectures (more layers, different sizes)
- 🎭 New activation functions (Tanh, Leaky ReLU)
- 🚀 Advanced optimizers (Adam, RMSprop)
- 📈 Learning rate scheduling

## 🤝 Contributing

We welcome contributions! 🎉

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. 💻 Make your changes
4. 🧪 Add tests if needed
5. 📝 Submit a pull request

## 📚 References

- [CS565600 Deep Learning](https://nthu-datalab.github.io/ml/index.html), National Tsing Hua University
- [Building a Neural Network from Scratch: Part 1](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/)
- [Building a Neural Network from Scratch: Part 2](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/)
- [Neural networks from scratch](https://developer.ibm.com/technologies/artificial-intelligence/articles/neural-networks-from-scratch), IBM Developer
- [The Softmax Function Derivative (Part 1)](https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/)

---

⭐ **Star this repository if you found it helpful!** ⭐
