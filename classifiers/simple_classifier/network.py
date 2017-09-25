from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(7)

dataset = numpy.loadtxt("data.csv", delimiter=",")
print(dataset.shape)
X = dataset[:760,0:8] # a total of 768, taking only 760.. remaining is for testing
Y = dataset[:760,8]

X_test = dataset[760:,0:8]

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=2, batch_size=100)

# Evaluate model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Summary
model.summary()

# Predict
n = model.predict(X_test)
print(n)