import numpy as np

class FeedForwardNN:
    def _init(self, n_input, n_hidden, n_output, learning_rate=0.01):  # Fixed __init_
        self.learning_rate = learning_rate
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(n_input, n_hidden) * 0.1
        self.bias_hidden = np.zeros(n_hidden)
        self.weights_hidden_output = np.random.randn(n_hidden, n_output) * 0.1
        self.bias_output = np.zeros(n_output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.final_input  # No activation for regression
        return self.final_output

    def backward(self, X, y, output):
        error = y - output
        output_gradient = -2 * error
        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, output_gradient)
        self.bias_output -= self.learning_rate * np.sum(output_gradient, axis=0)
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_gradient)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_gradient, axis=0)

    def fit(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)

# Main block
if _name_ == "_main":  # Fixed __name_
    X = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y = np.array([[0], [2], [4], [6], [8]], dtype=float)

    X /= np.max(X)
    y /= np.max(y)

    nn = FeedForwardNN(n_input=1, n_hidden=10, n_output=1, learning_rate=0.1)
    nn.fit(X, y, epochs=1000)

    predictions = nn.predict(X)
    print("Predictions:", predictions)
    print("Actual values:", y)
