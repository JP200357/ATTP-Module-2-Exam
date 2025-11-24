# Supervised Machine Learning Fashion-MNIST 
This project demonstrates a **feedforward neural network** implemented in TensorFlow/Keras to classify images from the **Fashion-MNIST dataset**. The dataset contains 70,000 grayscale images of fashion items across 10 categories.

## Project Structure

1.**Data Loading and Splitting**
-Training set: 60,000 images.
-Test set: 10,000 images.
-Validation set: 10% of training data (6,000 images)

2.**Data Preprocessing**
-Normalization: Pixel values scaled from 0–255 to 0–1.
-Flattening: 28×28 images converted to 784-dimensional vectors.
-One-hot encoding for labels.

3.**Model Architecture**
-Input layer: 784 neurons.
-Hidden layer: 128 neurons, ReLU activation.
-Output layer: 10 neurons, softmax activation.

4.**Training**
-Optimizer: Adam.
-Loss function: Categorical Crossentropy.
-Epochs: 15.
-Batch size: 128.
-Validation data: 6,000 images.

5.**Results**
-Training and validation accuracy improved steadily across epochs.
-Loss decreased consistently during training.

6.**Visualization**
Training and validation accuracy plot:

**Requirements**
-Python 3.8+
-TensorFlow 2.x
-NumPy
-Matplotlib

**Install dependencies:**
pip install tensorflow numpy matplotlib

**Run the Python script:**
Supervised_ML_Fashion_MNIST.ipynb

Observe the training process, validation accuracy, and final test accuracy.
