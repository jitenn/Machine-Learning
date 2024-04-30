import matplotlib.pyplot as plt 
# import tensorflow as tf
# keras = tf.keras
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the Pre-trained Model
model = load_model('mnist_model.h5')  

# Load and Preprocess Testing Data (same as in training)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0  
x_test = x_test.reshape(-1, 28*28)
y_test = to_categorical(y_test) 

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Predictions on a few images
predictions = model.predict(x_test[:5])
print(np.argmax(predictions, axis=1))  

# Confusion Matrix (with scikit-learn)
y_pred = model.predict(x_test) 
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred_classes) 

# Visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='binary') 
plt.show()
