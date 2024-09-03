import csv
from wine_quality_nn import SimpleNN

def load_data(filepath):
    """
    Load the wine dataset from a CSV file.

    :param filepath: Path to the CSV file containing the wine data.
    :return: List of data rows, each containing input features followed by the target output.
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return list(reader)

if __name__ == "__main__":
    # Load training data
    data = load_data('winequality-red.csv')

    # Initialize the neural network with 11 input features, 5 hidden neurons, and 1 output neuron
    model = SimpleNN(input_size=11, hidden_size=5, output_size=1)

    # Train the model
    model.train(train_data=data, epochs=100, learning_rate=0.01)
