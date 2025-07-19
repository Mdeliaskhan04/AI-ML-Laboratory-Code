import numpy as np
import matplotlib.pyplot as plt

# Input patterns (with bias term as -1 in last column)
X = np.array([
    [0.8, 0.5, 0.0, -1],
    [0.9, 0.7, 0.3, -1],
    [1.0, 0.8, 0.5, -1],
    [0.0, 0.2, 0.3, -1],
    [0.2, 0.1, 1.3, -1],
    [0.2, 0.7, 0.8, -1]
])

# Target outputs (1 for Class 1, 0 for Class 2)
y = np.array([1, 1, 1, 0, 0, 0])

def sign_activation(x):
    return 1 if x > 0 else 0

def train_perceptron(X, y, learning_rate=1, epochs=10):
    weights = np.zeros(X.shape[1])  # Initialize weights to zero
    
    for epoch in range(epochs):
        total_error = 0
        
        for i in range(len(X)):
            prediction = sign_activation(np.dot(X[i], weights))
            error = y[i] - prediction
            weights += 0.5 * learning_rate * error * X[i]
            total_error += error**2
        
        print(f"Epoch {epoch}: Weights = {np.round(weights, 4)}, Error = {total_error}")
        
        if total_error == 0:
            break
    
    return weights

# Train the perceptron
final_weights = train_perceptron(X, y)

# Test the perceptron
print("\nClassification Results:")
print("Pattern | Target | Output")
for i in range(len(X)):
    output = sign_activation(np.dot(X[i], final_weights))
    print(f"{X[i,:-1]} | {y[i]} | {output}")

# Plot the classified patterns
plt.figure(figsize=(8, 6))
for i in range(len(X)):
    color = 'red' if y[i] == 1 else 'blue'
    plt.scatter(X[i,0], X[i,1], c=color, s=100, 
                marker='*' if y[i] == 1 else 'o')
    
plt.title('Pattern Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()
