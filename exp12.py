import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("  Experiment 12: Dropout Regularization - Fashion-MNIST")
print("=" * 60)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255.0
X_test  = X_test  / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

def build_model_no_dropout():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10,  activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_with_dropout():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10,  activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

EPOCHS = 30
BATCH  = 128

print("\n  Training model WITHOUT Dropout...")
model_no_drop = build_model_no_dropout()
hist_no_drop  = model_no_drop.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH,
                                   validation_split=0.1, verbose=0)

print("  Training model WITH Dropout...")
model_drop = build_model_with_dropout()
hist_drop  = model_drop.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH,
                             validation_split=0.1, verbose=0)

loss_nd, acc_nd = model_no_drop.evaluate(X_test, y_test_cat, verbose=0)
loss_d,  acc_d  = model_drop.evaluate(X_test,    y_test_cat, verbose=0)

print("\n" + "=" * 52)
print("              Results Comparison")
print("=" * 52)
print(f"{'Model':<22} {'Test Acc (%)':>14} {'Test Loss':>12}")
print("-" * 50)
print(f"{'Without Dropout':<22} {acc_nd*100:>14.2f} {loss_nd:>12.4f}")
print(f"{'With Dropout':<22} {acc_d*100:>14.2f}  {loss_d:>12.4f}")

gap_nd = max(hist_no_drop.history['accuracy']) - max(hist_no_drop.history['val_accuracy'])
gap_d  = max(hist_drop.history['accuracy'])    - max(hist_drop.history['val_accuracy'])
print(f"\nOverfitting Gap (train-val acc):")
print(f"  Without Dropout : {gap_nd:.4f}")
print(f"  With Dropout    : {gap_d:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(hist_no_drop.history['accuracy'],     label='Train', color='royalblue')
axes[0,0].plot(hist_no_drop.history['val_accuracy'], label='Val',   color='orange')
axes[0,0].set_title('Without Dropout - Accuracy')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(hist_no_drop.history['loss'],     label='Train', color='royalblue')
axes[0,1].plot(hist_no_drop.history['val_loss'], label='Val',   color='orange')
axes[0,1].set_title('Without Dropout - Loss')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(hist_drop.history['accuracy'],     label='Train', color='forestgreen')
axes[1,0].plot(hist_drop.history['val_accuracy'], label='Val',   color='red')
axes[1,0].set_title('With Dropout - Accuracy')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(hist_drop.history['loss'],     label='Train', color='forestgreen')
axes[1,1].plot(hist_drop.history['val_loss'], label='Val',   color='red')
axes[1,1].set_title('With Dropout - Loss')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Experiment 12 - Dropout Regularization on Fashion-MNIST', fontweight='bold')
plt.tight_layout()
plt.savefig('exp12_dropout_comparison.png', dpi=100)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(hist_no_drop.history['val_accuracy'], label='Without Dropout', color='royalblue', linewidth=2)
plt.plot(hist_drop.history['val_accuracy'],    label='With Dropout',    color='forestgreen', linewidth=2)
plt.title('Experiment 12 - Validation Accuracy Comparison')
plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('exp12_val_accuracy.png', dpi=100)
plt.show()

print("\nPlots saved: exp12_dropout_comparison.png, exp12_val_accuracy.png")
