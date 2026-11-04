import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("  Experiment 4: Activation Functions Visualization")
print("=" * 55)

x = np.linspace(-10, 10, 500)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_fn(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def relu_deriv(x):
    return np.where(x > 0, 1.0, 0.0)

print(f"\n{'x':>6}  {'Sigmoid':>10}  {'Tanh':>10}  {'ReLU':>8}")
print("-" * 42)
for val in [-5, -2, -1, 0, 1, 2, 5]:
    print(f"{val:>6}  {sigmoid(val):>10.6f}  {tanh_fn(val):>10.6f}  {relu(val):>8.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Experiment 4 - Activation Functions', fontsize=14, fontweight='bold')

data = [(sigmoid(x), 'Sigmoid  σ(x) = 1/(1+e⁻ˣ)', 'royalblue'),
        (tanh_fn(x), 'Tanh  tanh(x)',               'forestgreen'),
        (relu(x),    'ReLU  max(0, x)',               'crimson')]

for ax, (vals, title, color) in zip(axes, data):
    ax.plot(x, vals, color=color, linewidth=2)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('Input x')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exp4_activations.png', dpi=100)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Experiment 4 - Derivatives of Activation Functions', fontsize=14, fontweight='bold')

deriv_data = [(sigmoid_deriv(x), "Sigmoid Derivative",  'royalblue'),
              (tanh_deriv(x),    "Tanh Derivative",     'forestgreen'),
              (relu_deriv(x),    "ReLU Derivative",     'crimson')]

for ax, (vals, title, color) in zip(axes, deriv_data):
    ax.plot(x, vals, color=color, linewidth=2, linestyle='--')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('Input x')
    ax.set_ylabel("f'(x)")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exp4_derivatives.png', dpi=100)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid(x),  label='Sigmoid', color='royalblue',   linewidth=2)
plt.plot(x, tanh_fn(x),  label='Tanh',    color='forestgreen', linewidth=2)
plt.plot(x, relu(x),     label='ReLU',    color='crimson',     linewidth=2)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.title('Experiment 4 - All Activations Overlaid')
plt.xlabel('Input x'); plt.ylabel('f(x)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exp4_overlay.png', dpi=100)
plt.show()

print("\nPlots saved: exp4_activations.png, exp4_derivatives.png, exp4_overlay.png")
