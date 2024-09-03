from wine_quality_nn import SimpleNN

def load_trained_model():
    """
    Initialize and return a trained neural network model.
    This function assumes that the weights are set manually or loaded from a file.
    """
    model = SimpleNN(input_size=11, hidden_size=5, output_size=1)
    # Manually set trained weights and biases if necessary (or assume retraining each time)
    return model

if __name__ == "__main__":
    # Load the trained model
    model = load_trained_model()

    # Define a test sample input for prediction
    test_sample = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]  # Example input

    # Predict the quality of the wine sample
    prediction = model.predict(test_sample)
    print(f'Predicted Wine Quality: {prediction}')
