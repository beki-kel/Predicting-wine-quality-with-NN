import random
class SimpleNN:
    """
    A simple feedforward neural network class to predict wine quality.
    This implementation does not use any external libraries like NumPy or Pandas.
    It includes a single hidden layer, ReLU activation, and manual backpropagation.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with random weights and biases.

        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layer.
        :param output_size: Number of output neurons.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases for input to hidden layer
        self.weights_input_hidden = [[random.random() for _ in range(hidden_size)] for _ in range(input_size)]
        self.bias_hidden = [random.random() for _ in range(hidden_size)]

        # Initialize weights and bias for hidden to output layer
        self.weights_hidden_output = [random.random() for _ in range(hidden_size)]
        self.bias_output = random.random()

    def relu(self, x):
        """
        ReLU activation function.
        Returns the input value if it's positive; otherwise, returns 0.
        """
        return max(0, x)

    def relu_derivative(self, x):
        """
        Derivative of the ReLU function.
        Returns 1 if the input is positive; otherwise, returns 0.
        """
        return 1 if x > 0 else 0

    def forward(self, inputs):
        """
        Perform the forward pass through the network.

        :param inputs: List of input values.
        :return: Output value and hidden layer outputs.
        """
        # Hidden layer calculations
        hidden_layer_activation = [
            sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(self.input_size)) + self.bias_hidden[j]
            for j in range(self.hidden_size)
        ]
        hidden_layer_output = [self.relu(x) for x in hidden_layer_activation]

        # Output layer calculations
        output = sum(hidden_layer_output[j] * self.weights_hidden_output[j] for j in range(self.hidden_size)) + self.bias_output
        return output, hidden_layer_output

    def backward(self, inputs, hidden_layer_output, predicted_output, actual_output, learning_rate):
        """
        Perform the backward pass (backpropagation) to adjust weights and biases.

        :param inputs: List of input values.
        :param hidden_layer_output: Output from the hidden layer.
        :param predicted_output: Output from the forward pass.
        :param actual_output: The true value for the output.
        :param learning_rate: Learning rate for weight and bias updates.
        """
        # Output layer error
        output_error = predicted_output - actual_output
        d_output = output_error

        # Hidden layer error
        d_hidden_layer = [
            d_output * self.weights_hidden_output[j] * self.relu_derivative(hidden_layer_output[j])
            for j in range(self.hidden_size)
        ]

        # Update weights and biases
        self.weights_hidden_output = [
            self.weights_hidden_output[j] - learning_rate * d_output * hidden_layer_output[j]
            for j in range(self.hidden_size)
        ]
        self.bias_output -= learning_rate * d_output

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] -= learning_rate * d_hidden_layer[j] * inputs[i]
            self.bias_hidden[j] -= learning_rate * d_hidden_layer[j]

    def train(self, train_data, epochs, learning_rate):
        """
        Train the neural network over a specified number of epochs.

        :param train_data: List of training data rows, where each row is a list of input features followed by the target output.
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for weight and bias updates.
        """
        for epoch in range(epochs):
            total_loss = 0
            for row in train_data:
                inputs = list(map(float, row[:-1]))
                actual_output = float(row[-1])
                predicted_output, hidden_layer_output = self.forward(inputs)
                total_loss += (predicted_output - actual_output) ** 2
                self.backward(inputs, hidden_layer_output, predicted_output, actual_output, learning_rate)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data)}')

    def predict(self, inputs):
        """
        Predict the output for given input data.

        :param inputs: List of input values.
        :return: Predicted output value.
        """
        predicted_output, _ = self.forward(inputs)
        return predicted_output