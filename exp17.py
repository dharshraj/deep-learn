import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

print("=" * 62)
print("  Experiment 17: Max Pooling vs Average Pooling - CIFAR-10")
print("=" * 62)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

def build_maxpool_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10,  activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_avgpool_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        AveragePooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        AveragePooling2D((2,2)),
        Conv2D(128,(3,3), activation='relu', padding='same'),
        AveragePooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10,  activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

EPOCHS = 20
BATCH  = 128

print("\n  Training CNN with Max Pooling...")
model_max  = build_maxpool_cnn()
model_max.summary()
hist_max   = model_max.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH,
                            validation_split=0.1, verbose=1)

print("\n  Training CNN with Average Pooling...")
model_avg  = build_avgpool_cnn()
hist_avg   = model_avg.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH,
                            validation_split=0.1, verbose=1)

loss_max, acc_max = model_max.evaluate(X_test, y_test_cat, verbose=0)
loss_avg, acc_avg = model_avg.evaluate(X_test, y_test_cat, verbose=0)

print("\n" + "=" * 52)
print("          Pooling Method Comparison")
print("=" * 52)
print(f"{'Method':<18} {'Test Acc (%)':>14} {'Test Loss':>12}")
print("-" * 48)
print(f"{'Max Pooling':<18} {acc_max*100:>14.2f} {loss_max:>12.4f}")
print(f"{'Average Pooling':<18} {acc_avg*100:>14.2f} {loss_avg:>12.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(hist_max.history['accuracy'],     label='Train', color='royalblue')
axes[0,0].plot(hist_max.history['val_accuracy'], label='Val',   color='orange')
axes[0,0].set_title('Max Pooling - Accuracy')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(hist_max.history['loss'],     label='Train', color='royalblue')
axes[0,1].plot(hist_max.history['val_loss'], label='Val',   color='orange')
axes[0,1].set_title('Max Pooling - Loss')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(hist_avg.history['accuracy'],     label='Train', color='forestgreen')
axes[1,0].plot(hist_avg.history['val_accuracy'], label='Val',   color='red')
axes[1,0].set_title('Average Pooling - Accuracy')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(hist_avg.history['loss'],     label='Train', color='forestgreen')
axes[1,1].plot(hist_avg.history['val_loss'], label='Val',   color='red')
axes[1,1].set_title('Average Pooling - Loss')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Experiment 17 - Max vs Average Pooling on CIFAR-10', fontweight='bold')
plt.tight_layout()
plt.savefig('exp17_pooling_comparison.png', dpi=100)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(hist_max.history['val_accuracy'], label='Max Pooling',     color='royalblue',   linewidth=2)
plt.plot(hist_avg.history['val_accuracy'], label='Average Pooling', color='forestgreen', linewidth=2)
plt.title('Experiment 17 - Validation Accuracy: Max vs Average Pooling')
plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('exp17_val_accuracy.png', dpi=100)
plt.show()

print("\nPlots saved: exp17_pooling_comparison.png, exp17_val_accuracy.png")
