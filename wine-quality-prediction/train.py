import csv
import random
from wine_quality_nn import SimpleNN

def normalizeData(dataset):
    """normalize the dataset to the range [0, 1]."""
    inputs = [row[:-1] for row in dataset]
    minVals = [min(column) for column in zip(*inputs)]
    maxVals = [max(column) for column in zip(*inputs)]

    normalized_dataset = []
    for row in dataset:
        normalized_input = [(x - minVals[i]) / (maxVals[i] - minVals[i]) for i, x in enumerate(row[:-1])]
        normalized_dataset.append(normalized_input + [row[-1]])
    
    return normalized_dataset, minVals, maxVals

def denormalize(value, min_val, max_val):
    """denormalize the value from [0, 1] to original range."""
    return value * (max_val - min_val) + min_val

def splitdataset(dataset, train_ratio=0.8):
    """this splits the dataset into training and test sets"""
    dataset = list(dataset)  # Ensure dataset is a list
    random.shuffle(dataset)
    split_index = int(len(dataset) * train_ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data

def load_data(filepath):
    
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip the header row
        for row in reader:
            data.append([float(value) for value in row])
    return data

#loading the dataset and split to train and test
dataset = load_data('./winequality-red.csv')
normalized_dataset, minVals, maxVals = normalizeData(dataset)
train_data, test_data = splitdataset(normalized_dataset, train_ratio=0.8)

#training with the data set
nn = SimpleNN(sizeofInput=2, sizeofHidden=3, sizeofOutput=1)
nn.train(train_data, epochs=100, learningrate=0.01)

predictions = []
actuals = []

for testRow in test_data:
    inputData = testRow[:-1]
    actualOutput = testRow[-1]
    predictedoutput = nn.predict(inputData)
    predictions.append(predictedoutput)
    actuals.append(actualOutput)

    print(f"Input: {inputData}, Actual Output: {actualOutput}, Predicted Output: {predictedoutput}")

# Evaluating the prediction
def meanSquarederror(predictions, actuals):
    """Calculate Mean Squared Error."""
    return sum((pred - actual) ** 2 for pred, actual in zip(predictions, actuals)) / len(predictions)

def meanAbsoluteerror(predictions, actuals):
    """Calculate Mean Absolute Error."""
    return sum(abs(pred - actual) for pred, actual in zip(predictions, actuals)) / len(predictions)

mse = meanSquarederror(predictions, actuals)
mae = meanAbsoluteerror(predictions, actuals)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")