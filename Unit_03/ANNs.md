# Artificial Neural Networks

## What is a fully connected artificial neural network (i.e., multilayer perceptron)?

A fully connected neural network, also known as a dense neural network or multilayer perceptron, is one of the simplest types of artificial neural networks consisting of layers and neurons. It is characterized by the way its layers are structured and interconnected.

A fully connected neural network typically consists of an input layer, one or more hidden layers, and an output layer. Each layer is composed of nodes, also known as neurons.

In a fully connected network, every neuron in one layer is connected to every neuron in the subsequent layer. This means that each neuron receives inputs from all neurons in the previous layer and sends outputs to all neurons in the next layer.

Fully connected networks can approximate any continuous function and are thus versatile for many tasks (e.g., regression or classification).

They make a good starting point for beginners in deep learning because they are straightforward to implement.

Fully connected neural networks do have some limitations. Due to the large number of connections, they can be computationally expensive and memory-intensive, especially as the number of neurons increases. They are also prone to overfitting, especially when the network is large and the amount of training data is limited. Techniques like dropout and regularization are often used to mitigate this issue.

## What is a neuron, and what math is involved?

The basic units of neural networks are neurons (i.e., nodes), which are typically organized into layers. Neurons are the building blocks of neural networks, enabling complex computations and learning patterns from data. They are also responsible for processing and transmitting information within the network.

Neurons receive multiple inputs, which are typically numerical values. These inputs can come from external data in the input layer or from other neurons in previous layers.

Each input is associated with a weight. These weights are adjustable parameters that determine the strength or importance of each input in the calculation.

A neuron may also have a bias term, which is an additional parameter added to the weighted sum of inputs. The bias allows the activation function to be shifted, providing more flexibility in learning.

After computing the weighted sum (products of the features and weights) of inputs plus the bias, the neuron applies an activation function. This function introduces non-linearity and determines the neuron's output.

Weights ($w_i$) and bias ($b$) terms are learned. Activation functions f(z) are not; they are “hyperparameters.”

### Neuron in Math Form

1. Multiply feature(s) and weight(s)
2. Take the sum of all the products of the features and weights
3. Add bias term

$z = \sum (w_i \cdot x_i) + b$

Where ($w_i$) are the weights, ($x_i$) are the inputs, and ($b$) is the bias.

4. Pass the signal through the activation function

$\text{output} = \text{activation}(z)$

## What are neural network hyperparameters?

Hyperparameters are parameters whose values are set (by a human) before starting the model training process. The choice of hyperparameters can significantly affect the neural network's performance and efficiency. Hyperparameters can interact with each other in complex ways, making it challenging to find the optimal configuration.

| Method    | Description |
| -------- | ------- |
| Data preprocessing | How you preprocess your data is important! |
| Number of layers | Determines how deep the neural network is. More layers can potentially capture more complex patterns but may also lead to overfitting. |
| Number of neurons per layer | Specifies how many neurons are in each layer. More neurons can increase the model's capacity to learn, but also increase computational complexity. |
| Learning rate | Controls the rate of update/step size for weights during each iteration of backpropagation. Usually between 0.01 and 0.0001. A small learning rate can lead to a longer training time, while a large learning rate can cause the model to converge too quickly to a suboptimal solution. |
| Batch size | Refers to the number of training examples utilized in one iteration. Smaller batch sizes can lead to noisier updates but may help generalization. |
| Weight initialization | Method to initialize the weights. Can be zero, one, random... |
| Activation function    | Function that calculates the output of a node based on the sum of the inputs times weights and bias. Can add nonlinearity. |
| Loss function    | What error would you like your neural network to learn to minimize? |
| Number of epochs| Defines how many times the learning algorithm will work through the entire training dataset. More epochs can improve learning, but may also increase the risk of overfitting. |
| Dropout rate | Specifies the probability of dropping out neurons during training to prevent overfitting. |
| Regularization coefficients | Add penalties to the loss function to discourage complex models (e.g., L1 or L2 regularization). |
| Optimizer | Determines the algorithm used to update the weights, such as SGD (Stochastic Gradient Descent), Adam, or RMSprop. |

Hyperparameter tuning is the process of finding the optimal set of hyperparameters that yield the best performance for a given model and dataset. This can be done through methods like grid search (i.e., exhaustively searching through a specified subset of hyperparameter space), random search (i.e., random search samples random combinations), or more advanced techniques like Bayesian optimization (i.e., models the hyperparameter space probabilistically). Hyperparameter tuning can be computationally expensive and time-consuming, especially for large models.

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

## How do neurons learn?

## The bias-variance trade-off

## Troubleshooting training a neural network
