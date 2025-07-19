import numpy as np

# Input patterns (flattened + bias term)
X = np.array([
    [1,0,0, 1,0,0, 1,1,1, -1],   # L
    [1,1,1, 0,1,0, 0,1,0, -1]    # I
])

# Targets
y = np.array([1, 0])  # L -> 1, I -> 0


def sign(x):
    return 1 if x > 0 else 0

def train_perceptron(X, y, learning_rate=1, epochs=20):
    weights = np.zeros(X.shape[1])
    
    for epoch in range(epochs):
        total_error = 0
        print(f"Epoch {epoch+1}:")
        
        for i in range(len(X)):
            output = sign(np.dot(X[i], weights))
            error = y[i] - output
            weights += learning_rate * error * X[i]
            total_error += error**2
            
            print(f" Input: {X[i][:-1]} | Target: {y[i]} | Output: {output} | Weights: {weights}")
        
        if total_error == 0:
            print("\nTraining converged.")
            break
    
    return weights

# Train perceptron
final_weights = train_perceptron(X, y)

# Testing
print("\nFinal Weights:", final_weights)
print("\nTesting on Inputs:")
for i in range(len(X)):
    result = sign(np.dot(X[i], final_weights))
    print(f" Input: {X[i][:-1]} | Target: {y[i]} | Output: {result}")
