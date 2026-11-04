import numpy as np

np.random.seed(42)

print("=" * 58)
print("  Experiment 6: Forward Propagation - Multilayer NN")
print("=" * 58)

X = np.array([[0.5, 0.1, 0.4],
              [0.9, 0.8, 0.2],
              [0.3, 0.6, 0.7],
              [0.1, 0.4, 0.9]])
y = np.array([[1], [0], [1], [0]])

print("\nInput X:\n", X)
print("\nLabels y:\n", y.T)

W1 = np.array([[ 0.2, -0.1,  0.4,  0.3],
               [ 0.5,  0.2, -0.3,  0.1],
               [-0.1,  0.4,  0.2, -0.2]])
b1 = np.array([[ 0.1,  0.0, -0.1,  0.2]])

W2 = np.array([[ 0.3,  0.1, -0.2],
               [-0.1,  0.4,  0.3],
               [ 0.2, -0.3,  0.1],
               [ 0.4,  0.2, -0.1]])
b2 = np.array([[-0.1,  0.2,  0.1]])

W3 = np.array([[ 0.5],
               [-0.3],
               [ 0.4]])
b3 = np.array([[0.1]])

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("\n" + "-" * 55)
print("  LAYER 1  (ReLU)  :  3 inputs -> 4 neurons")
print("-" * 55)
Z1 = X @ W1 + b1
A1 = relu(Z1)
print("Weighted Sum Z1:\n", np.round(Z1, 4))
print("\nReLU Output A1:\n", np.round(A1, 4))

print("\n" + "-" * 55)
print("  LAYER 2  (ReLU)  :  4 inputs -> 3 neurons")
print("-" * 55)
Z2 = A1 @ W2 + b2
A2 = relu(Z2)
print("Weighted Sum Z2:\n", np.round(Z2, 4))
print("\nReLU Output A2:\n", np.round(A2, 4))

print("\n" + "-" * 55)
print("  OUTPUT LAYER  (Sigmoid)  :  3 inputs -> 1 output")
print("-" * 55)
Z3 = A2 @ W3 + b3
A3 = sigmoid(Z3)
print("Weighted Sum Z3:\n", np.round(Z3, 4))
print("\nSigmoid Output A3:\n", np.round(A3, 4))

print("\n" + "=" * 52)
print("       Forward Propagation - Final Results")
print("=" * 52)
print(f"{'Sample':<8} {'Actual':>8} {'Network Output':>16} {'Predicted':>10}")
print("-" * 46)
for i in range(len(X)):
    predicted = 1 if A3[i][0] >= 0.5 else 0
    print(f"{'S'+str(i+1):<8} {y[i][0]:>8} {A3[i][0]:>16.6f} {predicted:>10}")

mse = np.mean((y - A3) ** 2)
print(f"\nMean Squared Error (MSE): {mse:.6f}")

total = W1.size + b1.size + W2.size + b2.size + W3.size + b3.size
print(f"Total Network Parameters: {total}")
