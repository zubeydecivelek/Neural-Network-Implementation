# Flower Classification with Neural Networks

This project implements a multi-layer neural network for classifying flower images from the Flowers15 dataset.

## Dependencies

This project requires the following Python libraries:

- numpy
- cv2 (OpenCV)
- keras
- matplotlib
- pickle
- pandas
- seaborn

### Creating the requirements.txt file:

1. Open a text editor (e.g., Notepad, Sublime Text).
2. List down all the required libraries mentioned above.
3. Save the file as requirements.txt in your project directory.

## Data Loading and Preprocessing

The code first loads flower images from three categories (train, validation, and test) within the flowers15 directory. These images are then converted to grayscale, resized to a fixed size (64x64 pixels), and normalized by dividing pixel values by 255.

## Visualizing Sample Images

A function `show_images` is used to visualize a few sample images from each dataset (training, validation, and test) to get a sense of the data distribution.

## Reshaping Data for Neural Network Input

The image data is reshaped into a format suitable for the neural network. Since we have grayscale images with a width and height of 64 pixels, the data is reshaped to have a single channel and a total of 64 * 64 elements per image.

## Neural Network Class with Early Stopping

The core of this project is the `NeuralNetworkWithEarlyStopping` class. This class defines a multi-layer neural network with the following functionalities:

- Initialization: Takes the number of neurons in each layer as input and allows specifying the activation function (default is 'relu').
- Activation Functions: Supports various activation functions, including 'relu', 'tanh', 'leaky_relu', 'sigmoid', and the option to use a custom activation function.
- Forward Pass: Propagates the input data through the network layers, calculating activations at each layer.
- Loss Function: Computes the softmax cross-entropy loss between the predicted outputs and the true labels.
- Backward Pass: Computes the gradients of the loss function with respect to the weights and biases using backpropagation.
- Update Parameters: Updates the network weights and biases based on the calculated gradients and a learning rate.
- Batch Normalization (Optional): Can be optionally enabled for potentially faster convergence and improved performance.
- Numerical Gradient Check: Performs a numerical gradient check to verify the correctness of backpropagation calculations.
- Training: Trains the network on the provided training data using mini-batch stochastic gradient descent with early stopping to prevent overfitting. Early stopping monitors validation loss and stops training if there is no improvement for a certain number of epochs (patience).
- Prediction: Generates class predictions for new input data.
- Evaluation: Evaluates the network's performance on a given dataset by calculating metrics such as accuracy, precision, recall, and F1-score.
- Plotting Loss and Accuracy: Plots the training and validation loss/accuracy curves over epochs to visualize the learning process.
- Saving and Loading Model: Saves the trained model parameters (weights and biases) to a file and allows loading them back for future use.
- Visualizing Weights: Visualizes the weights of each layer in the network to gain insights into how the network learned to represent features.
- Confusion Matrix: Calculates and plots the confusion matrix to understand the distribution of prediction errors across different classes.

## Dataset

The Flowers15 dataset is not included in this repository due to size constraints. You can download it from [Google Drive](https://drive.google.com/file/d/1zhP3F6anZEmOkmZZOfXN_soSxdtkK2Rn/view).

## Experiment Design and Results

The code performs experiments with different hyperparameter configurations (number of hidden layers, learning rate, and batch size) to evaluate the performance of the neural network. The results are reported in a Pandas DataFrame for easy comparison.

## Running the Code

1. Clone this repository (assuming you have Git installed).
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the Flowers15 dataset and place it in the `flowers15` directory.
4. Run the main script (e.g., `python main.py`).

The code will train the neural network, evaluate its performance, and potentially generate plots and visualizations based on the chosen configurations.
