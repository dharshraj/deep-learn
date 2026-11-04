import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

print("=" * 68)
print("  Experiment 19: CNN Hyperparameter Tuning - CIFAR-10")
print("=" * 68)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

configs = [
    {'filter_size': (3,3), 'num_filters': 16,  'label': '3×3 filters, 16 ch'},
    {'filter_size': (3,3), 'num_filters': 32,  'label': '3×3 filters, 32 ch'},
    {'filter_size': (3,3), 'num_filters': 64,  'label': '3×3 filters, 64 ch'},
    {'filter_size': (5,5), 'num_filters': 16,  'label': '5×5 filters, 16 ch'},
    {'filter_size': (5,5), 'num_filters': 32,  'label': '5×5 filters, 32 ch'},
    {'filter_size': (5,5), 'num_filters': 64,  'label': '5×5 filters, 64 ch'},
]

def build_cnn(filter_size, num_filters):
    model = Sequential([
        Conv2D(num_filters,     filter_size, activation='relu',
               padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(num_filters * 2, filter_size, activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(num_filters * 4, filter_size, activation='relu', padding='same'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(10,  activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

EPOCHS = 15
BATCH  = 128
results = {}

for cfg in configs:
    label = cfg['label']
    print(f"\n  Training: {label} ...")
    model = build_cnn(cfg['filter_size'], cfg['num_filters'])

    t0   = time.time()
    hist = model.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH,
                     validation_split=0.1, verbose=0)
    train_time = time.time() - t0

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    total_params = model.count_params()
    results[label] = {
        'hist': hist, 'test_acc': acc,
        'test_loss': loss, 'time': train_time,
        'params': total_params
    }
    print(f"  Test Acc: {acc*100:.2f}%  |  Time: {train_time:.1f}s  |  Params: {total_params:,}")

print("\n" + "=" * 78)
print("                     Hyperparameter Tuning Summary")
print("=" * 78)
print(f"{'Config':<26} {'Test Acc (%)':>13} {'Test Loss':>11} {'Time (s)':>10} {'Params':>10}")
print("-" * 75)
for label, r in results.items():
    print(f"{label:<26} {r['test_acc']*100:>13.2f} {r['test_loss']:>11.4f} "
          f"{r['time']:>10.1f} {r['params']:>10,}")

colors_3x3 = ['#1f77b4', '#2196F3', '#00BCD4']
colors_5x5 = ['#e74c3c', '#FF5722', '#FF9800']

labels_3x3 = [c['label'] for c in configs[:3]]
labels_5x5 = [c['label'] for c in configs[3:]]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, label in enumerate(labels_3x3):
    axes[0].plot(results[label]['hist'].history['val_accuracy'],
                 label=label, color=colors_3x3[i], linewidth=2)
axes[0].set_title('3×3 Filter Size - Validation Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

for i, label in enumerate(labels_5x5):
    axes[1].plot(results[label]['hist'].history['val_accuracy'],
                 label=label, color=colors_5x5[i], linewidth=2)
axes[1].set_title('5×5 Filter Size - Validation Accuracy')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Experiment 19 - CNN Hyperparameter Tuning on CIFAR-10', fontweight='bold')
plt.tight_layout()
plt.savefig('exp19_tuning_curves.png', dpi=100)
plt.show()

plt.figure(figsize=(11, 5))
labels = list(results.keys())
accs   = [results[l]['test_acc']*100 for l in labels]
bar_colors = colors_3x3 + colors_5x5
bars   = plt.bar(range(len(labels)), accs, color=bar_colors, edgecolor='black', width=0.6)
for bar, val in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', fontsize=9)
plt.xticks(range(len(labels)), labels, rotation=20, ha='right', fontsize=9)
plt.ylabel('Test Accuracy (%)'); plt.ylim(55, 80)
plt.title('Experiment 19 - Test Accuracy by Configuration')
plt.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig('exp19_accuracy_bar.png', dpi=100)
plt.show()

print("\nPlots saved: exp19_tuning_curves.png, exp19_accuracy_bar.png")
