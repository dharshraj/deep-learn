import numpy as np
import matplotlib.pyplot as plt

print("=" * 58)
print("  Experiment 7: Backpropagation - XOR Dataset")
print("=" * 58)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

np.random.seed(0)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))
lr = 0.5
epochs = 10000
losses = []

print("\nInitial Weights W1:\n", np.round(W1, 4))
print("\nInitial Weights W2:\n", np.round(W2, 4))
print("\nTraining for", epochs, "epochs...\n")

for epoch in range(epochs):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    loss = np.mean((y - A2) ** 2)
    losses.append(loss)

    dLoss_dA2 = -(y - A2)
    dA2_dZ2   = sigmoid_deriv(A2)
    delta2    = dLoss_dA2 * dA2_dZ2

    dW2 = A1.T @ delta2
    db2 = np.sum(delta2, axis=0, keepdims=True)

    dA1_dZ1 = sigmoid_deriv(A1)
    delta1   = (delta2 @ W2.T) * dA1_dZ1

    dW1 = X.T @ delta1
    db1 = np.sum(delta1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 1000 == 0:
        print(f"  Epoch {epoch+1:5d} | Loss: {loss:.6f}")

print("\nFinal Weights W1:\n", np.round(W1, 4))
print("\nFinal Weights W2:\n", np.round(W2, 4))

print("\n" + "=" * 52)
print("         Final Predictions After Training")
print("=" * 52)
print(f"{'Input':<10} {'Actual':>8} {'Output':>12} {'Predicted':>10}")
print("-" * 44)
for i in range(len(X)):
    A1p = sigmoid(X[i] @ W1 + b1)
    A2p = sigmoid(A1p @ W2 + b2)
    pred = 1 if A2p[0][0] >= 0.5 else 0
    print(f"{str(X[i]):<10} {y[i][0]:>8} {A2p[0][0]:>12.6f} {pred:>10}")

plt.figure(figsize=(9, 4))
plt.plot(losses, color='royalblue', linewidth=1.5)
plt.title('Experiment 7 - Backpropagation Loss Reduction (XOR)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exp7_backprop_loss.png', dpi=100)
plt.show()
print("\nPlot saved: exp7_backprop_loss.png")
