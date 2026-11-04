import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

print("=" * 55)
print("  Experiment 3: Multilayer NN - Iris Classification")
print("=" * 55)

iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y_cat = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42)

print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")
print(f"Classes: {iris.target_names}")

model = Sequential([
    Dense(16, input_shape=(4,), activation='relu'),
    Dense(12, activation='relu'),
    Dense(8,  activation='relu'),
    Dense(3,  activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                    validation_split=0.1, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy : {acc * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=iris.target_names))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='Train', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Validation', color='orange')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Validation', color='orange')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Experiment 3 - Iris Classification Training', fontweight='bold')
plt.tight_layout()
plt.savefig('exp3_iris_training.png', dpi=100)
plt.show()

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title('Experiment 3 - Confusion Matrix')
plt.savefig('exp3_iris_confusion.png', dpi=100)
plt.show()
print("Plots saved: exp3_iris_training.png, exp3_iris_confusion.png")
