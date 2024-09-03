import random
class SimpleNN:
    def __init__(self, sizeofInput, sizeofHidden, sizeofOutput):

        self.sizeofHidden = sizeofHidden
        self.sizeofInput = sizeofInput
        self.sizeofOutput = sizeofOutput

        self.weights_input_hidden = []
        for i in range(sizeofInput):
            row = []
            for j in range(sizeofHidden):
                row.append(random.random())
            self.weights_input_hidden.append(row)

        self.bias_hidden= []
        for j in range(sizeofHidden):
            self.bias_hidden.append(random.random())

        
        self.weightsHiddenOutput=[]
        for k in range(sizeofHidden)  :
              self.weightsHiddenOutput.append(random.random())  
            
        self.bias_output = random.random()
    
    """relu activation function"""
    def relu(self, x): 
        return max(0, x)

    """derivative of the relu function."""
    def reluDerivative(self, x):
        if x > 0:
            return 1
        else:
            return 0

    """this the func. that will do the forward pass in the network. 
    it takes the list of inupt values and retrun output value and hidden layer output"""
    def ForwardPropagation(self, inputs):
        # the calcs in the hidden layer
        hiddenlayerActivation = []
        for j in range(self.sizeofHidden):
            activation = 0
            for i in range(self.sizeofInput):
                activation += inputs[i] * self.weights_input_hidden[i][j]
            activation += self.bias_hidden[j]
            hiddenlayerActivation.append(activation)

        hiddenlayerOutput = [self.relu(x) for x in hiddenlayerActivation]

        # the calcs in the output layer
        output = 0
        for j in range(self.sizeofHidden):
            output += hiddenlayerOutput[j] * self.weightsHiddenOutput[j]
        output += self.bias_output

        return output, hiddenlayerOutput

    def BacwardPropagation(self, input, hiddenLayerOutput, predictedutput, actualOutput, learningrate):
        """Perform the BacwardPropagation pass (backpropagation) to adjust weights and biases."""
        # Output layer error
        outputError = predictedutput - actualOutput
        d_output = outputError

        # Hidden layer error
        d_hidden_layer = [
            d_output * self.weightsHiddenOutput[j] * self.reluDerivative(hiddenLayerOutput[j])
            for j in range(self.sizeofHidden)
        ]

        # Update weights and biases
        self.weightsHiddenOutput = [
            self.weightsHiddenOutput[j] - learningrate * d_output * hiddenLayerOutput[j]
            for j in range(self.sizeofHidden)
        ]
        self.bias_output -= learningrate * d_output

        for i in range(self.sizeofInput):
            for j in range(self.sizeofHidden):
                self.weights_input_hidden[i][j] -= learningrate * d_hidden_layer[j] * input[i]
            self.bias_hidden[j] -= learningrate * d_hidden_layer[j]

    def train(self, trainData, epochs, learningrate):
        """func. to Train the neural network for specified numbers of epochs"""
        for epoch in range(epochs):
            totaloss = 0
            for row in trainData:
                input = list(map(float, row[:-1]))
                actualOutput = float(row[-1])
                predictedutput, hiddenLayerOutput = self.ForwardPropagation(input)
                totaloss += (predictedutput - actualOutput) ** 2
                self.BacwardPropagation(input, hiddenLayerOutput, predictedutput, actualOutput, learningrate)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {totaloss / len(trainData)}')

    
    def predict(self, input, outputMin, outputMax):
        """ a func. to return for the predicted output value for the given input"""
        predictedutput, _ = self.ForwardPropagation(input)
        return predictedutput * (outputMax - outputMin) + outputMin