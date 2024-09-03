import random

class SimpleNN:
    def __init__(self, sizeofInput, sizeofHidden, sizeofOutput):
        self.sizeofHidden = sizeofHidden
        self.sizeofInput = sizeofInput
        self.sizeofOutput = sizeofOutput

        self.weights_input_hidden = [[random.random() for _ in range(sizeofHidden)] for _ in range(sizeofInput)]
        self.bias_hidden = [random.random() for _ in range(sizeofHidden)]

        self.weightsHiddenOutput = [random.random() for _ in range(sizeofHidden)]
        self.bias_output = random.random()
    
    """relu activation function"""
    def relu(self, x): 
        return max(0, x)

    """derivative of the relu function."""
    def reluDerivative(self, x):
        return 1 if x > 0 else 0

    """this the func. that will do the forward pass in the network. 
    it takes the list of inupt values and retrun output value and hidden layer output"""
    def ForwardPropagation(self, inputs):
        hiddenlayerActivation = []
        for j in range(self.sizeofHidden):
            activation = sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(self.sizeofInput))
            activation += self.bias_hidden[j]
            hiddenlayerActivation.append(activation)

        hiddenlayerOutput = [self.relu(x) for x in hiddenlayerActivation]

        output = sum(hiddenlayerOutput[j] * self.weightsHiddenOutput[j] for j in range(self.sizeofHidden))
        output += self.bias_output

        return output, hiddenlayerOutput

    def BackwardPropagation(self, input, hiddenLayerOutput, predicted_output, actualOutput, learningrate):
        """ this func. will do the BacwardPropagation pass to adjust weights and biases."""
        outputError = predicted_output - actualOutput
        d_output = outputError

        d_hidden_layer = [d_output * self.weightsHiddenOutput[j] * self.reluDerivative(hiddenLayerOutput[j]) for j in range(self.sizeofHidden)]

        self.weightsHiddenOutput = [self.weightsHiddenOutput[j] - learningrate * d_output * hiddenLayerOutput[j] for j in range(self.sizeofHidden)]
        self.bias_output -= learningrate * d_output

        for i in range(self.sizeofInput):
            for j in range(self.sizeofHidden):
                self.weights_input_hidden[i][j] -= learningrate * d_hidden_layer[j] * input[i]
        for j in range(self.sizeofHidden):
            self.bias_hidden[j] -= learningrate * d_hidden_layer[j]

    def train(self, trainData, epochs, learningrate):
        """ trains the neural network over a specified number of epochs."""
        for epoch in range(epochs):
            total_loss = 0
            for row in trainData:
                input = list(map(float, row[:-1]))
                actualOutput = float(row[-1])
                predicted_output, hiddenLayerOutput = self.ForwardPropagation(input)
                total_loss += (predicted_output - actualOutput) ** 2
                self.BackwardPropagation(input, hiddenLayerOutput, predicted_output, actualOutput, learningrate)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(trainData)}')

    def predict(self, inputs):
        """predicts the output for a given input."""
        if isinstance(inputs[0], list):
            return [self.ForwardPropagation(row)[0] for row in inputs]
        else:
            return self.ForwardPropagation(inputs)[0]
