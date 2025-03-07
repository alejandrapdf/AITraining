# AI Training Trials with Perceptron Class

This project includes a series of experiments and trials to demonstrate different ways to train a **Perceptron** model for binary classification tasks using different datasets. It leverages the `Perceptron` class defined earlier and includes examples on training with small logical datasets, real-world datasets like MNIST, and experimenting with both stochastic and batch training.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Part 1: Logical Operations](#part-1-logical-operations)
3. [Part 2: Training on MNIST Data](#part-2-training-on-mnist-data)
4. [Part 3: Batch Training on MNIST Data](#part-3-batch-training-on-mnist-data)
5. [Visualization](#visualization)
6. [Dependencies](#dependencies)
7. [License](#license)

## Project Overview

This project demonstrates how the **Perceptron** algorithm can be used for simple binary classification tasks, including both basic logical operations and real-world datasets like MNIST. The goal is to showcase different approaches to training the perceptron and evaluate its performance. It also explores the effect of different training methods, such as **stochastic gradient descent** and **batch training**.

### Three Main Parts:
1. **Part 1**: Testing the perceptron on basic logical operations (e.g., AND gate).
2. **Part 2**: Training and testing the perceptron on MNIST dataset for digit recognition (specifically recognizing the digit 7).
3. **Part 3**: Batch training on the MNIST dataset and comparing performance.

## Part 1: Logical Operations

In **Part 1**, we train the perceptron on a basic logical operation, such as the **AND gate**. This demonstrates how the perceptron can learn simple logical relationships.

```python
# Initialize perceptron with 3 inputs
p = Perceptron(3)

# Input data for logical operations (e.g., AND gate)
logic_input = [
    np.array([1, 0, 0]),
    np.array([1, 0, 1]),
    np.array([1, 1, 0]),
    np.array([1, 1, 1])
]

# Labels for the logical operation (AND)
logic_label = [0, 0, 0, 1]

# Train and test perceptron
p.test(logic_input, logic_label)
p.train(logic_input, logic_label)
p.test(logic_input, logic_label)
```

In this section, the perceptron is trained on simple data and tested for accuracy in learning the AND operation.

## Part 2: Training on MNIST Data

**Part 2** uses the **MNIST dataset**, a large collection of handwritten digits. The perceptron is trained to recognize a specific target digit (e.g., 7).

```python
# Initialize perceptron for MNIST with 785 inputs (784 features + 1 bias)
p = Perceptron(785)

# Load training and testing data from MNIST dataset
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

# Target digit (e.g., 7)
target_digit = 7

# Prepare input and label data
train_input = [np.append([1], d[1:]) for d in train_data]
train_label = [int(d[0] == target_digit) for d in train_data]
test_input = [np.append([1], d[1:]) for d in test_data]
test_label = [int(d[0] == target_digit) for d in test_data]

# Train perceptron
p.test(test_input, test_label)
p.train(train_input, train_label)
p.test(test_input, test_label)
```

This section demonstrates how the perceptron learns to classify handwritten digits as either the target digit (7) or not (binary classification).

## Part 3: Batch Training on MNIST Data

In **Part 3**, we implement **batch training** to optimize the weight updates over all training samples. Batch training calculates the average error for each iteration and adjusts weights based on the cumulative update.

```python
# Initialize perceptron for MNIST with 785 inputs (784 features + 1 bias)
p = Perceptron(785)

# Load training and testing data from MNIST dataset
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

# Prepare input and label data for target digit 7
train_input = [np.append([1], d[1:]) for d in train_data]
train_label = [int(d[0] == target_digit) for d in train_data]
test_input = [np.append([1], d[1:]) for d in test_data]
test_label = [int(d[0] == target_digit) for d in test_data]

# Test, train using batch method, and evaluate accuracy
p.test(test_input, test_label)
p.train_batch(train_input, train_label)
p.test(test_input, test_label)
```

Batch training helps in improving the learning process by considering all training samples in each iteration. This method often leads to better convergence and stability in training.

## Visualization

The perceptron weights can be visualized as an image for MNIST data, where each weight corresponds to a pixel value in a 28x28 grid (since MNIST images are 28x28 pixels). This helps in understanding the learned patterns.

```python
# Visualization of learned weights after training
fig = plt.figure(figsize=(4,4))
data = p.weights[1:].reshape(28, 28)
vis = train_input[0][1:].reshape(28, 28)
plt.imshow(vis)
plt.show()
```

This will display an image of the learned weights for the first training sample (if it’s related to a digit like 7). You can observe how the model's weights correspond to the features it has learned.

## Dependencies

- **NumPy**: This project relies on NumPy for array manipulations and mathematical operations.
  
  Install using:
  ```bash
  pip install numpy
  ```

- **Matplotlib**: For visualizing the data and the learned weights.

  Install using:
  ```bash
  pip install matplotlib
  ```

## License

This project is open-source and available under the MIT License.
