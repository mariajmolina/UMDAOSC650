# Welcome to AOSC650!

### What is the goal of this course?

My goal is that you leave this course with a sufficient foundation on neural networks (NNs), where

- you can understand related scientific papers and talks, 
- you can determine whether they would be useful for your research question, and 
- you can determine which class of NNs would be best suited for your problem.

### Learning outcomes:

- Students will be able to prepare their physical science data for neural network workflows and choose preprocessing methods based on the characteristics of their data and problem.
- Students will be able to describe neural network architectures of various types and identify which neural network and type of learning would be appropriate for certain applications in the physical sciences.
- Students will be able to detect when neural networks have overfit to the training data and identify when learning has gone wrong by describing what neural networks have learned and performing robust hyperparameter grid searching and optimization.

I’ve purposefully structured this course so that you will get out of it what you put into it. 

Lectures will help, your class project will help, but asking questions and pushing yourself to apply NNs to your real-world data will be key to your expertise.

### Your Final Course Project

Entails the application of a neural network to a scientific question of your interest using a physical science dataset.

Grading Rubric: 

1. Application and justification of your neural network (e.g., unsupervised or supervised, classification or regression, image or pixel or time series, complexity).
2. Application and justification for your data preprocessing (e.g., data dimensionality, normalization and/or standardization, class imbalance handling, input variable choices, label creation).
3. Explanation and justification of your training process and hyperparameter choices (e.g., number of layers, loss function, cross validation, learning rate, regularization, class weights).
4. Explanation and justification of your NN skill assessment (e.g., properly assessing skill of classification or regression NN, skill assessment of class imbalance situations, are your clusters meaningful?).
5. Explanation of what your NN learned (e.g., test data/prediction visualization, explainable AI methods).

What if my NN never learns??

Identification and justification of what went wrong and how you could potentially move forward.

# Introduction to Soup of Terminology

You have likely heard several terms used interchangeably, such as machine learning and artificial intelligence, but many commonly used terms have important distinctions.

### Computer Science

Computer Science serves as the foundational discipline that covers the principles and technologies underlying all computational processes (e.g., software, hardware, data structure, networking and security, and theory of computation).

### Artificial Intelligence

Artificial Intelligence is a subset of computer science focused on creating systems that can perform tasks that typically require human intelligence (e.g., perception, language understanding, theory of intelligence, problem solving, and decision making).

### Machine Learning

Machine Learning is a subset of AI that involves the development of algorithms and models that allow computers to learn from data (e.g., automation, pattern recognition, and ``data-driven'').

### Deep Learning

Deep Learning is a specialized subset of machine learning that uses neural networks with many layers (hence "deep") to model complex patterns in data (e.g., computer vision, generative AI).

### How Does Not Using Machine/Deep Learning Look?

When you do not use machine learning, you typically rely on traditional programming methods where:

- Explicitly program instructions and rules for the computer to follow.
- Traditional programs do not adapt or improve over time unless manually updated.
- For complex tasks, creating explicit rules can be challenging and time-consuming.
  
