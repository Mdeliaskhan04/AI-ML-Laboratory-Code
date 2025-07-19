import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D visualization

class DiscretePerceptron:
    def __init__(self, learning_rate=1.0):
        self.weights = None
        self.lr = learning_rate
    
    def activate(self, x):
        """Discrete activation function (unipolar)"""
        return 1 if x > 0 else 0
    
    def train(self, X, y, max_epochs=10, verbose=True):
        """
        Train the perceptron on given data
        Args:
            X: Input features with bias term (n_samples x n_features)
            y: Target labels (1 or 0)
            max_epochs: Maximum training iterations
            verbose: Whether to print training progress
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights to zero
        
        for epoch in range(max_epochs):
            total_error = 0
            weight_updates = np.zeros_like(self.weights)
            
            for i in range(n_samples):
                # Calculate net input
                net_input = np.dot(X[i], self.weights)
                
                # Get prediction
                prediction = self.activate(net_input)
                
                # Calculate error
                error = y[i] - prediction
                total_error += error**2
                
                # Update weights (accumulate updates)
                weight_updates += 0.5 * self.lr * error * X[i]
            
            # Apply weight updates after each epoch
            self.weights += weight_updates
            
            if verbose:
                print(f"Epoch {epoch + 1}:")
                print(f"  Weights: {np.round(self.weights, 4)}")
                print(f"  Total Error: {total_error}")
                print("-" * 40)
            
            # Early stopping if no error
            if total_error == 0:
                print("Converged!")
                break
    
    def predict(self, X):
        """Predict class labels for input samples"""
        net_inputs = np.dot(X, self.weights)
        return np.array([self.activate(x) for x in net_inputs])

# Input data preparation
def prepare_data():
    """Create the dataset with bias term"""
    # Original features (first 3 columns)
    features = np.array([
        [0.8, 0.5, 0.0],
        [0.9, 0.7, 0.3],
        [1.0, 0.8, 0.5],
        [0.0, 0.2, 0.3],
        [0.2, 0.1, 1.3],
        [0.2, 0.7, 0.8]
    ])
    
    # Add bias term (-1) as last column
    X = np.hstack([features, -np.ones((6, 1))])
    
    # Target labels (1 for Class 1, 0 for Class 2)
    y = np.array([1, 1, 1, 0, 0, 0])
    
    return X, y

def visualize_results(X, y, weights, title):
    """3D visualization of the classification results"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot class 1 (original class 1)
    class1 = X[y == 1]
    ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], 
               c='r', marker='o', s=100, label='Class 1')
    
    # Plot class 2 (original class 2)
    class2 = X[y == 0]
    ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], 
               c='b', marker='^', s=100, label='Class 2')
    
    # Create decision boundary plane
    xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    zz = (-weights[3] - weights[0]*xx - weights[1]*yy) / weights[2]
    
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title(title)
    plt.legend()
    plt.show()

def main():
    # Prepare data
    X, y = prepare_data()
    print("Input Data with Bias Term:")
    print(np.hstack([X, y.reshape(-1, 1)]))
    print("\n" + "="*60 + "\n")
    
    # Create and train perceptron
    perceptron = DiscretePerceptron(learning_rate=1.0)
    perceptron.train(X, y)
    
    # Make predictions
    predictions = perceptron.predict(X)
    
    # Print final results
    print("\nFinal Results:")
    print("Index | Features (1-3) | Bias | Target | Prediction")
    for i in range(len(X)):
        print(f"{i+1:3}  | {X[i,0]:.1f} {X[i,1]:.1f} {X[i,2]:.1f} | {X[i,3]:3} | {y[i]:3} | {predictions[i]:3}")
    
    # Visualize decision boundary
    visualize_results(X[:, :-1], y, perceptron.weights, 
                    "Perceptron Classification with Decision Plane")

if __name__ == "__main__":
    main()