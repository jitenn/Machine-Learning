# Handwritten digits
# pip install keras tensorflow

from tensorflow import keras

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam 

# 1. Load and Prepare Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing (normalize, reshape if needed)
x_train = x_train / 255.0  
x_test = x_test / 255.0  
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1,  28*28)

# 2. Define the Model Architecture
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,))) # Hidden Layer
model.add(Dense(10, activation='softmax'))  # Output layer (classes 0-9)

# 3. Compile the Model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 4. Train the Model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 5. Evaluate 
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

model.save('mnist_model.h5')
