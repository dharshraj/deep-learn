import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1])

np.random.seed(42)
weights = np.zeros(2)
bias = 0.0
learning_rate = 0.1
epochs = 10

def step_function(x):
    return 1 if x >= 0.5 else 0

print("=" * 55)
print("    Experiment 1: Single-Layer Perceptron - AND Gate")
print("=" * 55)

for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)
        error = y[i] - prediction
        total_error += abs(error)
        weights += learning_rate * error * X[i]
        bias    += learning_rate * error
    print(f"Epoch {epoch+1:2d} | Weights: {weights} | Bias: {bias:.4f} | Total Error: {total_error}")

print("\n" + "=" * 45)
print("           Final Predictions")
print("=" * 45)
print(f"{'Input':<12} {'Actual':>8} {'Predicted':>10} {'Match':>6}")
print("-" * 40)
all_correct = True
for i in range(len(X)):
    out = np.dot(X[i], weights) + bias
    pred = step_function(out)
    match = "Yes" if pred == y[i] else "No"
    if pred != y[i]:
        all_correct = False
    print(f"{str(X[i]):<12} {y[i]:>8} {pred:>10}  {match:>6}")

print(f"\nFinal Weights : {weights}")
print(f"Final Bias    : {bias:.4f}")
print(f"All Correct   : {all_correct}")
