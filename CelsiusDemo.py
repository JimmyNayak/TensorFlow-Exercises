import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=5200, verbose=False)
print("Finished training the model")

# plt.xlabel('Epoch Number')
# plt.ylabel("Loss Magnitude")
# plt.plot(history.history['loss'])
# plt.show()

print(model.predict([100.0]))

print("These are the layer variables: {}".format(l0.get_weights()))

#  Train the new data model

# l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
# l1 = tf.keras.layers.Dense(units=4)
# l2 = tf.keras.layers.Dense(units=1)
# model = tf.keras.Sequential([l0, l1, l2])
# model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
# model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
# print("Finished training the model")
# print(model.predict([100.0]))
# print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(
#     model.predict([100.0])))
# print("These are the l0 variables: {}".format(l0.get_weights()))
# print("These are the l1 variables: {}".format(l1.get_weights()))
# print("These are the l2 variables: {}".format(l2.get_weights()))
