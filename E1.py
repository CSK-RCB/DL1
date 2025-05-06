import numpy as np

# Training data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 0],
    [0, 1],
    [3, 1]
])
y = np.array([1, 1, 1, -1, -1, -1])  # Changed 0s to -1s

weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.1

# Training loop
for epoch in range(10):
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        pred = np.sign(z)  # Changed from step to sign
        if pred != y[i]:   # Update only on misclassification
            weights += learning_rate * y[i] * X[i]
            bias += learning_rate * y[i]

# Prediction
predictions = np.sign(np.dot(X, weights) + bias)

print("Predicted labels:", predictions)
print("Actual labels:   ", y)
