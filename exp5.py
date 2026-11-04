import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("=" * 58)
print("  Experiment 5: ReLU vs Sigmoid vs Tanh - Breast Cancer")
print("=" * 58)

data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Features: {X.shape[1]}  |  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

def build_model(activation):
    model = Sequential([
        Dense(64, input_shape=(30,), activation=activation),
        Dense(32, activation=activation),
        Dense(16, activation=activation),
        Dense(1,  activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

activations = ['relu', 'sigmoid', 'tanh']
histories   = {}
results     = {}

for act in activations:
    print(f"\n  Training with {act.upper()} activation...")
    tf.random.set_seed(42)
    model = build_model(act)
    hist  = model.fit(X_train, y_train, epochs=50, batch_size=32,
                      validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    histories[act] = hist
    results[act]   = {'test_loss': loss, 'test_acc': acc}
    print(f"  Test Accuracy: {acc * 100:.2f}%  |  Test Loss: {loss:.4f}")

print("\n" + "=" * 52)
print("         Activation Comparison Summary")
print("=" * 52)
print(f"{'Activation':<12} {'Test Accuracy (%)':>18} {'Test Loss':>12}")
print("-" * 46)
for act in activations:
    print(f"{act.upper():<12} {results[act]['test_acc']*100:>18.2f} {results[act]['test_loss']:>12.4f}")

colors = {'relu': 'crimson', 'sigmoid': 'royalblue', 'tanh': 'forestgreen'}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = [
    ('accuracy',     'Training Accuracy'),
    ('val_accuracy', 'Validation Accuracy'),
    ('loss',         'Training Loss')
]

for ax, (metric, title) in zip(axes, metrics):
    for act in activations:
        ax.plot(histories[act].history[metric],
                label=act.upper(), color=colors[act], linewidth=2)
    ax.set_title(title); ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('val_', '').capitalize())
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Experiment 5 - Activation Functions on Breast Cancer Dataset',
             fontweight='bold')
plt.tight_layout()
plt.savefig('exp5_activation_comparison.png', dpi=100)
plt.show()

plt.figure(figsize=(6, 4))
accs = [results[act]['test_acc'] * 100 for act in activations]
bars = plt.bar([a.upper() for a in activations], accs,
               color=[colors[a] for a in activations], edgecolor='black', width=0.5)
for bar, val in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', fontsize=11)
plt.title('Test Accuracy by Activation Function')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 100)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('exp5_accuracy_bar.png', dpi=100)
plt.show()

print("\nPlots saved: exp5_activation_comparison.png, exp5_accuracy_bar.png")
