import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, Activation)
from tensorflow.keras.utils import to_categorical

print("=" * 65)
print("  Experiment 21: Batch Normalization Effect - CIFAR-10")
print("=" * 65)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

def build_no_bn():
    model = Sequential([
        Conv2D(32,  (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64,  (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_with_bn():
    model = Sequential([
        Conv2D(32,  (3,3), padding='same', input_shape=(32,32,3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2)),

        Conv2D(64,  (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

EPOCHS = 25
BATCH  = 128

print("\n  Training CNN WITHOUT Batch Normalization...")
model_no_bn = build_no_bn()
model_no_bn.summary()
hist_no_bn  = model_no_bn.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH,
                               validation_split=0.1, verbose=1)

print("\n  Training CNN WITH Batch Normalization...")
model_bn  = build_with_bn()
model_bn.summary()
hist_bn   = model_bn.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH,
                          validation_split=0.1, verbose=1)

loss_no, acc_no = model_no_bn.evaluate(X_test, y_test_cat, verbose=0)
loss_bn, acc_bn = model_bn.evaluate(X_test,    y_test_cat, verbose=0)

print("\n" + "=" * 55)
print("         Batch Normalization Comparison")
print("=" * 55)
print(f"{'Model':<28} {'Test Acc (%)':>14} {'Test Loss':>10}")
print("-" * 55)
print(f"{'Without Batch Norm':<28} {acc_no*100:>14.2f} {loss_no:>10.4f}")
print(f"{'With Batch Norm':<28}    {acc_bn*100:>14.2f} {loss_bn:>10.4f}")

threshold = 0.65
def epoch_to_reach(hist, threshold):
    for i, v in enumerate(hist.history['val_accuracy']):
        if v >= threshold:
            return i + 1
    return None

ep_no = epoch_to_reach(hist_no_bn, threshold)
ep_bn = epoch_to_reach(hist_bn,    threshold)
print(f"\nEpoch to reach {threshold*100:.0f}% val accuracy:")
print(f"  Without BN : {'Not reached' if ep_no is None else ep_no}")
print(f"  With BN    : {'Not reached' if ep_bn is None else ep_bn}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(hist_no_bn.history['accuracy'],     label='Train', color='royalblue')
axes[0,0].plot(hist_no_bn.history['val_accuracy'], label='Val',   color='orange')
axes[0,0].set_title('Without Batch Norm - Accuracy')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(hist_no_bn.history['loss'],     label='Train', color='royalblue')
axes[0,1].plot(hist_no_bn.history['val_loss'], label='Val',   color='orange')
axes[0,1].set_title('Without Batch Norm - Loss')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(hist_bn.history['accuracy'],     label='Train', color='forestgreen')
axes[1,0].plot(hist_bn.history['val_accuracy'], label='Val',   color='red')
axes[1,0].set_title('With Batch Norm - Accuracy')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(hist_bn.history['loss'],     label='Train', color='forestgreen')
axes[1,1].plot(hist_bn.history['val_loss'], label='Val',   color='red')
axes[1,1].set_title('With Batch Norm - Loss')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Experiment 21 - Batch Normalization on CIFAR-10', fontweight='bold')
plt.tight_layout()
plt.savefig('exp21_batchnorm_comparison.png', dpi=100)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(hist_no_bn.history['val_accuracy'], label='Without BN', color='royalblue',   linewidth=2)
plt.plot(hist_bn.history['val_accuracy'],    label='With BN',    color='forestgreen', linewidth=2)
plt.title('Experiment 21 - Val Accuracy: With vs Without Batch Norm')
plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('exp21_val_accuracy.png', dpi=100)
plt.show()

print("\nPlots saved: exp21_batchnorm_comparison.png, exp21_val_accuracy.png")
