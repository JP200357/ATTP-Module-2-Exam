# Fashion-MNIST Supervised Classification

This project demonstrates a **feedforward neural network** implemented in TensorFlow/Keras to classify images from the **Fashion-MNIST dataset**. The dataset contains 70,000 grayscale images of fashion items across 10 categories.

## Project Structure

1. **Data Loading and Splitting**
```python
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
Training set: 60,000 images

Test set: 10,000 images

Validation set: 10% of training data (6,000 images)

Data Preprocessing

Normalization: Pixel values scaled from 0–255 to 0–1.

Flattening: 28×28 images converted to 784-dimensional vectors.

One-hot encoding for labels.

Model Architecture

python
Copy code
model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
Input layer: 784 neurons

Hidden layer: 128 neurons, ReLU activation

Output layer: 10 neurons, softmax activation

Training

Optimizer: Adam

Loss function: Categorical Crossentropy

Epochs: 15

Batch size: 128

Validation data: 6,000 images

Example of training results:

yaml
Copy code
Epoch 1/15 - val_accuracy: 0.8408
Epoch 10/15 - val_accuracy: 0.8883
Epoch 15/15 - val_accuracy: 0.8910
Results

Final Test Accuracy: 0.8838

Training and validation accuracy improved steadily across epochs.

Loss decreased consistently during training.

Visualization
Training and validation accuracy plot:

python
Copy code
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
Requirements
Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib

How to Run
Install dependencies:

bash
Copy code
pip install tensorflow numpy matplotlib
Run the Python script:

bash
Copy code
python fashion_mnist_nn.py
Observe the training process, validation accuracy, and final test accuracy.
