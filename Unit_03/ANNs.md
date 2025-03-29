# Artificial Neural Networks

### What is a fully connected artificial neural network (i.e., multilayer perceptron)?

Artificial neural networks are machine learning models consisting of layers and neurons. The basic units are neurons (i.e., nodes), which are typically organized into layers.

### What is a neuron, and what math is involved?

### What are neural network hyperparameters?

### Common Activation Functions

- **Sigmoid Activation Function**

The Sigmoid function maps input values to a range between 0 and 1. It is commonly used in binary classification problems.

$\sigma(x) = \frac{1}{1 + e^{-x}}$

- **Hyperbolic Tangent (Tanh) Activation Function**

The Tanh function maps input values to a range between -1 and 1. It is similar to the sigmoid function but outputs zero-centered values.

$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

- **Rectified Linear Unit (ReLU) Activation Function**

The ReLU function is widely used due to its simplicity and effectiveness in handling the vanishing gradient problem.

$f(x) = \max(0, x)$

- **Leaky ReLU Activation Function**

Leaky ReLU is a variant of ReLU that allows a small, non-zero gradient when the input is negative, addressing the "dying ReLU" problem.

$If (x > 0), then (f(x) = x).$

$If (x \leq 0), then (f(x) = \alpha x).$

Where ($\alpha$) is a small constant.

- **Parametric ReLU (PReLU) Activation Function**

PReLU is similar to Leaky ReLU but with a learnable parameter (\alpha).

$If (x > 0), then (f(x) = x).$

$If (x \leq 0), then (f(x) = \alpha x).$

- **Exponential Linear Unit (ELU) Activation Function**

The ELU function aims to improve the learning characteristics by having negative values that push mean activations closer to zero.

$If (x > 0), then (f(x) = x).$

$If (x \leq 0), then (f(x) = \alpha (e^x - 1)).$

- **Softmax Activation Function**

The Softmax function is typically used in the output layer of a classification network to convert raw scores into probabilities.

$\sigma(x)i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$

Where ($x_i$) is the input to the ($i$)th neuron and ($n$) is the number of neurons in the layer.

- **Swish Activation Function**

Swish is a newer activation function that has been found to outperform ReLU in certain scenarios.

$f(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}$

- **Gaussian Activation Function**

The Gaussian activation function is used in some specialized neural network architectures and is defined by a bell-shaped curve.

$f(x) = e{-x^2}$

- **Softplus Activation Function**

The Softplus function is a smooth approximation of the ReLU function and is continuously differentiable.

$f(x) = \log(1 + e^x)$

- **Binary Step Activation Function**

The Binary Step function is used in simple threshold models and outputs either 0 or 1.

$If (x \geq 0), then (f(x) = 1).$

$If (x < 0), then (f(x) = 0).$

- **Hard Sigmoid Activation Function**

The Hard Sigmoid is a computationally efficient approximation of the sigmoid function.

$If (x \leq -2.5), then (f(x) = 0).$

$If (-2.5 < x < 2.5), then (f(x) = \frac{x}{5} + 0.5).$

$If (x \geq 2.5), then (f(x) = 1).$

- **Maxout Activation Function**

The Maxout function is a generalization of ReLU and can learn activation functions.

$f(x) = \max(w_1^T x + b_1, w_2^T x + b_2)$

Where ($w_1$, $w_2$) are weights and ($b_1$, $b_2$) are biases.

- **Mish Activation Function**

The Mish function is a smooth, non-monotonic function that has been shown to improve performance in some deep learning tasks.

$f(x) = x \cdot \tanh(\log(1 + e^x))$

- **SELU (Scaled Exponential Linear Unit) Activation Function**

The SELU function is designed to self-normalize the network, maintaining mean and variance close to zero and one, respectively.

$If (x > 0), then (f(x) = \lambda x).$

$If (x \leq 0), then (f(x) = \lambda \alpha (e^x - 1)).$

Where ($\lambda$) and ($\alpha$) are predefined constants.

- **Gaussian Error Linear Unit (GELU) Activation Function**

GELU is an activation function used in recent state-of-the-art models, such as BERT. It combines properties of both the ReLU and Gaussian distributions to provide smooth, non-linear activation.

$f(x) = x \cdot \Phi(x)$

Where ( $\Phi$ ) is the cumulative distribution function (CDF) of the Gaussian distribution. The GELU function can be approximated as:

$f(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$

### How do neurons learn?

### The bias-variance trade-off

### Troubleshooting training a neural network
