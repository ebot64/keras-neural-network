# keras-neural-network

### Title: Pima Indians Diabetes Database : Classification Problem

# Neural Network with Keras

## Import Libraries


```python
# first neural network with keras
import tensorflow as tf
```


```python
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

## Load Data


```python
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
```

## Define Keras Model


```python
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## Compile Keras Model


```python
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Fit Keras Model


```python
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
```

    Epoch 1/150
    77/77 [==============================] - 1s 1ms/step - loss: 2.4194 - accuracy: 0.5221
    Epoch 2/150
    77/77 [==============================] - 0s 1ms/step - loss: 1.2206 - accuracy: 0.5482
    Epoch 3/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.9537 - accuracy: 0.5690
    Epoch 4/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.8730 - accuracy: 0.5794
    Epoch 5/150
    ............
    ............
    Epoch 145/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5413 - accuracy: 0.6979
    Epoch 146/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5373 - accuracy: 0.7188
    Epoch 147/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5393 - accuracy: 0.7135
    Epoch 148/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5336 - accuracy: 0.7070
    Epoch 149/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5367 - accuracy: 0.7070
    Epoch 150/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5318 - accuracy: 0.7109
    

    <keras.callbacks.History at 0x1c92d68ff10>



## Evaluate Keras Model


```python
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
```

    24/24 [==============================] - 0s 938us/step - loss: 0.5844 - accuracy: 0.6758
    Accuracy: 67.58
    

## Make Predictions


```python
# make probability predictions with the model
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]

# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)

for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
```

    24/24 [==============================] - 0s 855us/step
    24/24 [==============================] - 0s 1ms/step
    [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0] => 1 (expected 1)
    [1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0] => 0 (expected 0)
    [8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0] => 1 (expected 1)
    [1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0] => 0 (expected 0)
    [0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0] => 0 (expected 1)
    

    C:\Users\ebot6\AppData\Local\Temp\ipykernel_24372\964902141.py:10: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
    


```python

```
