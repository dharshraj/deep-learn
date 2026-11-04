import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))
lr = 0.5
epochs = 10000
losses = []

print("=" * 50)
print("  Experiment 2: Feedforward NN for XOR Problem")
print("=" * 50)
print("Architecture: 2 --> 4 --> 1  (Sigmoid activation)")
print("\nTraining...")

for epoch in range(epochs):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    loss = np.mean((y - A2) ** 2)
    losses.append(loss)

    dA2 = -(y - A2)
    dZ2 = dA2 * sigmoid_deriv(A2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_deriv(A1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 2000 == 0:
        print(f"  Epoch {epoch+1:5d} | Loss: {loss:.6f}")

print("\n" + "=" * 52)
print("             XOR Final Predictions")
print("=" * 52)
print(f"{'Input':<10} {'Actual':>8} {'Raw Output':>12} {'Rounded':>8}")
print("-" * 42)
for i in range(len(X)):
    A1_p = sigmoid(X[i] @ W1 + b1)
    A2_p = sigmoid(A1_p @ W2 + b2)
    print(f"{str(X[i]):<10} {y[i][0]:>8} {A2_p[0][0]:>12.6f} {round(A2_p[0][0]):>8}")

plt.figure(figsize=(8, 4))
plt.plot(losses, color='royalblue', linewidth=1.5)
plt.title('Experiment 2 - XOR Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exp2_xor_loss.png', dpi=100)
plt.show()
print("\nLoss curve saved as exp2_xor_loss.png")
