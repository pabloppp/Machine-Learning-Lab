from keras.layers import Dense
from keras.models import Sequential
import numpy as np

X = np.matrix([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.matrix([1.0, 0.0, 0.0, 1.0]).T  # XOR

model = Sequential()
model.add(Dense(100, input_shape=(2,), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile('adam', 'binary_crossentropy', ['accuracy'])
model.fit(X, y, epochs=5000)

prediction = model.predict(X)
print(prediction)
