import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %matplotlib inline

# limit on size of vocabualry dictionary used in tweets
vocab_size = 500

data = pd.read_csv('Tweets.csv')
data.head()

# tokenizing of words in tweets
tok = Tokenizer(num_words=vocab_size, split=' ')
tok.fit_on_texts(data['text'].values)
X = tok.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

# extraction of sentiment category
categories = pd.get_dummies(data['airline_sentiment'])
labels = categories.keys()
Y = categories.values

# splitting into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print('train_features shape: ', X_train.shape)
print('test_features shape: ', X_test.shape)
print('train_labels shape: ', Y_train.shape)
print('test_labels shape: ', Y_test.shape)

# definition of model
model = Sequential()

# exercise 1
# Add layers to the model:
# - Embedding - it should get vectors of dictionary size lenght (vocab_size) on the input and transform them into vectors of the length equal to 32
# - 1 LSTM layer with number of units equal to 10
# - Dens - a base of classification (how many outputs it should have?)
# - learning process should be based on function starty categorical_crossentropy
# - choose 'sgd' as a method for model optimization
# - model should return accuracy metric (categorical_accuracy)

# ------------------------------------------------------------------------

# exercise 2 - zamien
# Change model optimization method to 'adam'.
# Compare results with those obtained with 'sgd' and explain differences.
# ------------------------------------------------------------------------

# exercise 3
# Add additional LSTM layer with number of units equal to 10.
# Perform learning process with 'adam' and 'sgd' methods.

## -- beginning of your solution

model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32))
model.add(tf.keras.layers.LSTM(10, return_sequences=True))
model.add(tf.keras.layers.LSTM(10))
model.add(tf.keras.layers.Dense(units=len(labels), activation="sigmoid"))

loss = tf.keras.losses.CategoricalCrossentropy()
predictions = model(X_train).numpy()
tf.nn.sigmoid(predictions).numpy()
loss(Y_train, predictions).numpy()

optimizer = tf.keras.optimizers.SGD()
optimizer2 = tf.keras.optimizers.Adam()

metrics = tf.keras.metrics.CategoricalAccuracy()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print(model.summary())

## -- end of your solution


# Add:
# - network learning on X_train, Y_train with parameters: batch_size = 16 and number of epochs = 5
# - accuracy checking on test data: X_test,Y_test

## -- beginning of your solution

model.fit(X_train, Y_train, batch_size=16, epochs=5)
model.evaluate(X_test, Y_test)

## -- end of your solution

# prediction on exemplary tweets


def predict(tweet):
    padded_tweet = pad_sequences(tok.texts_to_sequences([tweet]), maxlen=X.shape[1])
    scores = model.predict(padded_tweet)[0]
    index = np.argmax(scores)
    print(f'Tweet:\"{tweet}\"')
    print(f'predicted sentiment: {labels[index]}, confidence: {scores[index]}\n')


# expected prediction: negative
predict("@united been up since 4am cheers for this delay and then cancellation of the flight")
# expected prediction: positive
predict("@united Terrific. Many thanks. Looking forward to being back on UA tomorrow. Had a great flight up to Vancouver.")
# expected prediction: neutral
predict("Dallas, Texas to Marrakesh, Morocco for only $442 roundtrip with @FlySWISS & @united.")

