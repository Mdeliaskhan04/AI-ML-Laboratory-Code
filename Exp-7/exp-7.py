import numpy as np
pattern_L = np.array([
    1, 0, 0,
    1, 0, 0,
    1, 1, 1
])
pattern_I = np.array([
    0, 1, 0,
    0, 1, 0,
    0, 1, 0
])

target_L = -1
target_I = 1

W = pattern_L * target_L + pattern_I * target_I
b = 0

def sign_with_threshold(x, thresh=1):
    if x > thresh:
        return 1
    elif x < -thresh:
        return -1
    else:
        return 0

def take_input():
    print("Enter your 3x3 pattern row-wise (space separated 0 or 1):")
    user_input = []
    for i in range(3):
        row = input(f"Row {i+1}: ").strip().split()
        if len(row) != 3 or not all(x in ['0', '1'] for x in row):
            print("Invalid input. Enter exactly 3 binary digits (0 or 1).")
            return None
        user_input.extend([int(x) for x in row])
    return np.array(user_input)

def predict(X, threshold=1):
    net_input = np.dot(W, X) + b
    output = sign_with_threshold(net_input, threshold)
    return output

def classify(output):
    if output == -1:
        return "Pattern matched as: L"
    elif output == 1:
        return "Pattern matched as: I"
    else:
        return "Pattern does not match L or I."

def main():
    X = take_input()
    if X is not None:
        result = predict(X, threshold=1)
        print("\nNetwork Output (O):", result)
        print(classify(result))

if __name__ == "__main__":
    main()
