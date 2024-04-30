import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load your pre-trained model (assuming you saved it)
model = load_model('mnist_model.h5')  

# Load the MNIST testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ... (Optional preprocessing - if you had any)
x_test = x_test / 255.0  # Assuming you normalized during training
x_test = x_test.reshape(-1, 28*28)
y_test = to_categorical(y_test) 

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Predictions on a few images
predictions = model.predict(x_test[:5])
print(np.argmax(predictions, axis=1))  

# Basic Confusion Matrix (without scikit-learn)
confusion_matrix = np.zeros((10, 10))  # Initialize a 10x10 matrix

for i in range(len(y_test)):
    true_class = int(np.argmax(y_test[i]))  # Convert index to integer
    predicted_class = int(np.argmax(model.predict(x_test[i:i+1])[0]))
    confusion_matrix[true_class, predicted_class] += 1 

print(confusion_matrix)
