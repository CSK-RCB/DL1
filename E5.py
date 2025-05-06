import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# Load and visualize data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Train:", X_train.shape, y_train.shape, "| Test:", X_test.shape, y_test.shape)

# Show first 10 digits
fig, axs = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axs.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.show()

# Normalize and reshape
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Build model
model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=10, validation_split=0.2, validation_freq=5)

# Predict and evaluate
train_preds = model.predict(X_train).argmax(axis=1)
test_preds = model.predict(X_test).argmax(axis=1)

print("\nModel Summary:")
model.summary()

print("\nTraining Classification Report:")
print(classification_report(y_train, train_preds))

print("\nTest Classification Report:")
print(classification_report(y_test, test_preds))
