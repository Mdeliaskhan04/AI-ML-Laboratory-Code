import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def train_perceptron(X, y, eta=0.1, epochs=10000):
    X = np.insert(X, 0, -1, axis=1)
    w = np.random.rand(X.shape[1])
    for epoch in range(epochs):
        for i in range(len(X)):
            net = np.dot(w, X[i])
            output = sigmoid(net)
            error = y[i] - output
            delta = eta * error * sigmoid_derivative(output)
            w = w + delta * X[i]
    return w

def test_perceptron(X, w):
    X = np.insert(X, 0, -1, axis=1)
    outputs = []
    for i in range(len(X)):
        net = np.dot(w, X[i])
        output = sigmoid(net)
        outputs.append(output)
    return outputs

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

targets = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0])
}

for gate, y in targets.items():
    print(f"\n{gate} Gate:")
    w = train_perceptron(X, y, eta=0.5, epochs=10000)
    outputs = test_perceptron(X, w)
    print("Final Weights:", w)
    for i, out in enumerate(outputs):
        print(f"Input: {X[i]} → Output: {out:.4f} (Target: {y[i]})")
