import csv
from wine_quality_nn import SimpleNN

def load_data(filepath):
    import csv

    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=';')  # Specify the correct delimiter
        next(reader)  # Skip the header row
        for row in reader:
            data.append([float(value) for value in row])
    
    # Normalize data
    max_values = [max(col) for col in zip(*data)]
    min_values = [min(col) for col in zip(*data)]
    normalized_data = []
    for row in data:
        normalized_row = [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x, min_val, max_val in zip(row, min_values, max_values)]
        normalized_data.append(normalized_row)
    
    return normalized_data

if __name__ == "__main__":
    # Load and normalize training data
    data = load_data('./winequality-red.csv')

    # Initialize the neural network with 11 input features, two hidden layers with 8 and 4 neurons, and 1 output neuron
    model = SimpleNN(input_size=11, hidden_size=5, output_size=1)

    # Train the model
    model.train(train_data=data, epochs=100, learning_rate=0.01)