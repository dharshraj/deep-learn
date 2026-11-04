import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("  Experiment 10: SGD vs RMSprop vs Adam - Fashion-MNIST")
print("=" * 60)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test  = X_test  / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

def build_model(optimizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Dense(10,  activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

optimizers = {
    'SGD':     tf.keras.optimizers.SGD(learning_rate=0.01),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam':    tf.keras.optimizers.Adam(learning_rate=0.001)
}

histories = {}
results   = {}

for name, opt in optimizers.items():
    print(f"\n  Training with {name}...")
    tf.random.set_seed(42)
    model = build_model(opt)
    hist  = model.fit(X_train, y_train_cat, epochs=20, batch_size=128,
                      validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    histories[name] = hist
    results[name]   = {'test_loss': loss, 'test_acc': acc}
    print(f"  Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

print("\n" + "=" * 52)
print("           Optimizer Comparison Summary")
print("=" * 52)
print(f"{'Optimizer':<12} {'Test Accuracy (%)':>18} {'Test Loss':>12}")
print("-" * 46)
for name in optimizers:
    print(f"{name:<12} {results[name]['test_acc']*100:>18.2f} {results[name]['test_loss']:>12.4f}")

colors = {'SGD': 'crimson', 'RMSprop': 'forestgreen', 'Adam': 'royalblue'}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

metrics = [
    ('loss',         'Training Loss'),
    ('val_loss',     'Validation Loss'),
    ('val_accuracy', 'Validation Accuracy')
]

for ax, (metric, title) in zip(axes, metrics):
    for name in optimizers:
        ax.plot(histories[name].history[metric],
                label=name, color=colors[name], linewidth=2)
    ax.set_title(title); ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('val_', '').capitalize())
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Experiment 10 - Optimizer Comparison on Fashion-MNIST', fontweight='bold')
plt.tight_layout()
plt.savefig('exp10_optimizer_comparison.png', dpi=100)
plt.show()

plt.figure(figsize=(6, 4))
accs = [results[n]['test_acc']*100 for n in optimizers]
bars = plt.bar(list(optimizers.keys()), accs,
               color=[colors[n] for n in optimizers], edgecolor='black', width=0.5)
for bar, val in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.2f}%', ha='center', fontsize=11)
plt.title('Test Accuracy by Optimizer')
plt.ylabel('Accuracy (%)'); plt.ylim(75, 92)
plt.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig('exp10_accuracy_bar.png', dpi=100)
plt.show()

print("\nPlots saved: exp10_optimizer_comparison.png, exp10_accuracy_bar.png")
