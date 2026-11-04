import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("=" * 60)
print("  Experiment 11: Deep Feedforward NN - MNIST Digits")
print("=" * 60)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test  = X_test  / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.suptitle('Experiment 11 - MNIST Sample Images', fontweight='bold')
plt.tight_layout()
plt.savefig('exp11_samples.png', dpi=100)
plt.show()

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(10,  activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train_cat, epochs=20, batch_size=128,
                    validation_split=0.1, verbose=1)

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy : {acc * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history.history['accuracy'],     label='Train', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='orange')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Val',   color='orange')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Experiment 11 - MNIST Training Curves', fontweight='bold')
plt.tight_layout()
plt.savefig('exp11_training_curves.png', dpi=100)
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title('Experiment 11 - Confusion Matrix')
plt.savefig('exp11_confusion.png', dpi=100)
plt.show()

print("\nPlots saved: exp11_samples.png, exp11_training_curves.png, exp11_confusion.png")
