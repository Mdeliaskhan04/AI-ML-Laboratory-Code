import numpy as np
import matplotlib.pyplot as plt

class PatternClassifier:
    def __init__(self, input_size, learning_rate=1.0):
        self.weights = np.random.randn(input_size) * 0.01
        self.lr = learning_rate
    
    def bipolar_activate(self, x):
        """Bipolar activation function (returns 1 or -1)"""
        return 1 if x > 0 else -1
    
    def train(self, X, y, epochs=10, verbose=True):
        """Train the classifier"""
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # Forward pass
                net_input = np.dot(X[i], self.weights)
                prediction = self.bipolar_activate(net_input)
                
                # Calculate error
                error = y[i] - prediction
                total_error += abs(error)
                
                # Update weights
                self.weights += 0.5 * self.lr * error * X[i]
            
            errors.append(total_error)
            if verbose:
                print(f"Epoch {epoch+1}: Weights = {np.round(self.weights, 4)}, Error = {total_error}")
            
            if total_error == 0:
                print("Converged!")
                break
        return errors
    
    def predict(self, X):
        """Predict class labels"""
        return np.array([self.bipolar_activate(np.dot(x, self.weights)) for x in X])

# Define the 'L' and 'I' patterns (9 features + bias)
# 1 = pixel on, -1 = pixel off
L_pattern = np.array([
    1, 0, 0,  # | 
    1, 0, 0,  # |
    1, 1, 1   # _
])

I_pattern = np.array([
    1, 1, 1,  # _
    0, 1, 0,  # |
    1, 1, 1   # _
])

# Create dataset with bias term (-1)
X = np.array([
    np.append(L_pattern, -1),  # L pattern with bias
    np.append(I_pattern, -1)   # I pattern with bias
])
y = np.array([-1, 1])  # -1 for L, 1 for I

# Initialize and train classifier
classifier = PatternClassifier(input_size=10)
errors = classifier.train(X, y, epochs=10)

# Test with training data
predictions = classifier.predict(X)
print("\nTraining Results:")
print("Pattern | Target | Prediction")
print("---------------------------")
print(f"   L    |   -1   |   {predictions[0]}")
print(f"   I    |    1   |   {predictions[1]}")

# Test with distorted patterns (4 variations)
distorted_patterns = [
    np.array([1,0,0, 1,0,0, 1,0,0, -1]),  # Vertical line
    np.array([1,1,1, 0,0,0, 1,1,1, -1]),  # Horizontal bars
    np.array([1,0,1, 1,0,1, 1,1,1, -1]),  # L with right pixels
    np.array([1,1,1, 1,1,1, 1,1,1, -1])   # All on
]

print("\nDistorted Pattern Tests:")
for i, pattern in enumerate(distorted_patterns):
    pred = classifier.bipolar_activate(np.dot(pattern, classifier.weights))
    print(f"Pattern {i+1}: Prediction = {pred} ({'I' if pred == 1 else 'L'})")

# Plot training error
plt.figure()
plt.plot(range(1, len(errors)+1), errors, 'o-')
plt.xlabel('Epoch')
plt.ylabel('Total Error')
plt.title('Training Error Convergence')
plt.grid()
plt.show()