import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

print("=" * 65)
print("  Experiment 14: No Reg vs Dropout vs L2 - Fashion-MNIST")
print("=" * 65)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255.0
X_test  = X_test  / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

def build_no_reg():
    m = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Dense(10,  activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_dropout():
    m = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64,  activation='relu'),
        Dropout(0.2),
        Dense(10,  activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_l2():
    reg = regularizers.l2(0.001)
    m = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu', kernel_regularizer=reg),
        Dense(128, activation='relu', kernel_regularizer=reg),
        Dense(64,  activation='relu', kernel_regularizer=reg),
        Dense(10,  activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

EPOCHS = 30
BATCH  = 128

models_config = [
    ('No Regularization', build_no_reg),
    ('Dropout',           build_dropout),
    ('L2 (λ=0.001)',      build_l2)
]

histories = {}
results   = {}

for name, builder in models_config:
    print(f"\n  Training: {name}...")
    model = builder()
    hist  = model.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH,
                      validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    histories[name] = hist
    results[name]   = {'test_loss': loss, 'test_acc': acc}
    print(f"  Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

print("\n" + "=" * 58)
print("              Regularization Comparison Summary")
print("=" * 58)
print(f"{'Model':<22} {'Test Acc (%)':>14} {'Test Loss':>12} {'Overfit Gap':>12}")
print("-" * 62)
for name, _ in models_config:
    gap = (max(histories[name].history['accuracy']) -
           max(histories[name].history['val_accuracy']))
    print(f"{name:<22} {results[name]['test_acc']*100:>14.2f} "
          f"{results[name]['test_loss']:>12.4f} {gap:>12.4f}")

colors = {'No Regularization': 'crimson', 'Dropout': 'royalblue', 'L2 (λ=0.001)': 'forestgreen'}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

metrics = [
    ('accuracy',     'Training Accuracy'),
    ('val_accuracy', 'Validation Accuracy'),
    ('val_loss',     'Validation Loss')
]

for ax, (metric, title) in zip(axes, metrics):
    for name, _ in models_config:
        ax.plot(histories[name].history[metric],
                label=name, color=colors[name], linewidth=2)
    ax.set_title(title); ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('val_', '').capitalize())
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('Experiment 14 - Regularization Comparison on Fashion-MNIST', fontweight='bold')
plt.tight_layout()
plt.savefig('exp14_regularization_comparison.png', dpi=100)
plt.show()

plt.figure(figsize=(7, 4))
names = [n for n, _ in models_config]
accs  = [results[n]['test_acc']*100 for n in names]
bars  = plt.bar(names, accs, color=[colors[n] for n in names],
                edgecolor='black', width=0.5)
for bar, val in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', fontsize=11)
plt.title('Test Accuracy by Regularization Method')
plt.ylabel('Accuracy (%)'); plt.ylim(80, 92)
plt.xticks(fontsize=9); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('exp14_accuracy_bar.png', dpi=100)
plt.show()

print("\nPlots saved: exp14_regularization_comparison.png, exp14_accuracy_bar.png")
