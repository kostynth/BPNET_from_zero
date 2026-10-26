import numpy as np
import matplotlib.pyplot as plt

def generate_data(n: int) -> np.ndarray: # Generate data:
    x = np.linspace(0, 1, n)  # Generate n equally spaced points from 0 to 1
    x = x.reshape(len(x), 1)  # Reshape x to (n, 1)
    y = np.sin(2 * np.pi * x)  # Compute sin of x
    return x, y  # Return generated data

class NN:
    def __init__(self, n_input: int, n_hidden: int, n_output: int, num_layers: int) -> None:
        """
        n_input: input dimension
        n_hidden: number of hidden layer neurons
        n_output: output dimension
        num_layers: number of hidden layers
        """
        self.n_input = n_input  # input layer dimension
        self.n_hidden = n_hidden  # number of hidden layer neurons
        self.n_output = n_output  # output layer dimension
        self.num_layers = num_layers  # number of hidden layers
        self.w = []  # weight list
        self.b = []  # bias list
        self.gradw = []  # weight gradient list
        self.gradb = []  # bias gradient list
        self.deltas = []  # errors for each layer
        # Initialize weights
        self.w.append(np.random.randn(n_input, n_hidden))  # weights from input layer to first hidden layer
        for i in range(num_layers - 1):
            self.w.append(np.random.randn(n_hidden, n_hidden))  # weights from hidden layer to hidden layer
        self.w.append(np.random.randn(n_hidden, n_output))  # weights from last hidden layer to output layer
        # Initialize biases
        self.b.append(np.zeros((1, n_hidden)))  # bias of first hidden layer
        for i in range(num_layers - 1):
            self.b.append(np.zeros((1, n_hidden)))  # hidden layer bias
        self.b.append(np.zeros((1, n_output)))  # output layer bias

    def activation(self, x: np.ndarray) -> np.ndarray: # Activation function:
        return np.tanh(x)  # Use tanh as activation function

    def activation_deriv(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.power(self.activation(x), 2)  # Derivative of activation function

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = []  # Store weighted sum for each layer
        self.a = []  # Store activation value for each layer
        self.z.append(x)  # weighted sum of input layer
        self.a.append(x)  # activation value of input layer

        for i in range(self.num_layers):
            self.z.append(np.dot(self.a[i], self.w[i]) + self.b[i])  # Compute weighted sum for each layer
            self.a.append(self.activation(self.z[i + 1]))  # Compute activation value for each layer
        return self.a[-1]  # Return activation value of output layer

    def mse(self, x: np.ndarray, y: np.ndarray) -> float: # MSE loss function
        return np.mean(np.power(self.predict(x) - y, 2))  # Compute mean squared error

    def backward(self, x: np.ndarray, y: np.ndarray) -> None:

        self.gradw = []  # weight gradient list
        self.gradb = []  # bias gradient list
        self.deltas = []  # errors for each layer

        self.deltas.append(self.a[-1] - y)  # Compute output layer error

        self.gradw.append(np.dot(self.a[-2].T, self.deltas[-1]))  # Compute output layer weight gradient
        self.gradb.append(np.sum(self.deltas[-1], axis=0, keepdims=True))  # Compute output layer bias gradient

        for i in range(self.num_layers - 1, 0, -1):
            self.deltas.append(np.dot(self.deltas[-1], self.w[i].T) * self.activation_deriv(self.z[i]))  # Compute hidden layer error
            self.gradw.append(np.dot(self.a[i - 1].T, self.deltas[-1]))  # Compute hidden layer weight gradient
