import numpy as np
import matplotlib.pyplot as plt
# Parameters
init_weight_scale = 0.001
C = 1

# Input matrix: 6 samples, 3 features + bias
x = np.array([[0.8, 0.9, 1.0, 0.0, 0.2, 0.2],
              [0.5, 0.7, 0.8, 0.2, 0.1, 0.7],
              [0.0, 0.3, 0.5, 0.3, 1.3, 0.8],
              [-1, -1, -1, -1, -1, -1]])

# Target outputs
da = np.array([1, 1, 1, 0, 0, 0])

# Initialize weights
W = np.zeros(4)
WW = [W.copy()]  # To track weight changes

niter = 10
nSample = x.shape[1]
error_history = []

print('epoch      error')

# Activation function: discrete unipolar
def signh(h):
    return np.where(h > 0, 1, 0)

for i in range(niter):
    h = W @ x
    y = signh(h)
    e = da - y

    # Update weights
    f_d = 0.5 * C * e
    delta = f_d @ x.T
    W = W + delta
    WW.append(W.copy())

    # Calculate error
    error = np.sum(e**2) / nSample
    error_history.append(error)

    print(f'{i+1:4d} \t {error:.6f}')

WW = np.array(WW)

print('\nTraining Results')
print("Final Weights after 10 epochs:\n", W)
print('\nInput Patterns:')
print(x[:3].T)
print('\nPredicted Output:')
print(y)
print('\nError:')
print(e)

# Plot 1: Error over epochs
plt.figure(1)
plt.plot(range(1, niter+1), error_history, linewidth=2)
plt.xlim(1, niter)
plt.ylim(0, 0.5)
plt.title('Error Convergence Curve')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')

# Plot 2: Weight evolution
plt.figure(2)
iterations = range(1, niter+2)
plt.plot(iterations, WW[:, 3], '-^', label='w0 (bias)')
plt.plot(iterations, WW[:, 0], '-*', label='w1')
plt.plot(iterations, WW[:, 1], '-+', label='w2')
plt.plot(iterations, WW[:, 2], '-o', label='w3')
plt.xlim(1, niter+1)
plt.ylim(-2, 2)
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.title('Weight Updates per Epoch')
plt.legend()

plt.show()
