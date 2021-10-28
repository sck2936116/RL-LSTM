import numpy as np
from time import time
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
      print(e)

max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
inputs = Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = Embedding(max_features, 128)(inputs)
# Add 2 bidirectional LSTMs
x = LSTM(256, return_sequences=True)(x)
x = LSTM(256, return_sequences=True)(x)
x = LSTM(256, return_sequences=True)(x)
# Add a classifier
outputs = Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
start = time()
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
end = time()
print('Time taken for trainning:', end-start)