import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("=" * 60)
print("  Experiment 9: Binary Cross-Entropy - Pima Diabetes")
print("=" * 60)

try:
    df = pd.read_csv('diabetes.csv')
    print(f"Dataset loaded from CSV  |  Shape: {df.shape}")
except FileNotFoundError:
    print("diabetes.csv not found. Using synthetic placeholder data.")
    np.random.seed(42)
    n = 768
    df = pd.DataFrame({
        'Pregnancies':             np.random.randint(0, 17, n),
        'Glucose':                 np.random.randint(0, 200, n),
        'BloodPressure':           np.random.randint(0, 122, n),
        'SkinThickness':           np.random.randint(0, 99, n),
        'Insulin':                 np.random.randint(0, 846, n),
        'BMI':                     np.round(np.random.uniform(0, 67, n), 1),
        'DiabetesPedigreeFunction':np.round(np.random.uniform(0.08, 2.42, n), 3),
        'Age':                     np.random.randint(21, 82, n),
        'Outcome':                 np.random.randint(0, 2, n)
    })

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

print(f"Positive cases: {y.sum()}  |  Negative cases: {(y==0).sum()}")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, input_shape=(8,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.1, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc * 100:.2f}%")

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history.history['accuracy'],     label='Train', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='orange')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Val',   color='orange')
axes[1].set_title('Binary Cross-Entropy Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Experiment 9 - Pima Diabetes Classification', fontweight='bold')
plt.tight_layout()
plt.savefig('exp9_training_curves.png', dpi=100)
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap='Blues')
plt.title('Experiment 9 - Confusion Matrix')
plt.savefig('exp9_confusion.png', dpi=100)
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='royalblue', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], 'k--', linewidth=1)
plt.title('Experiment 9 - ROC Curve')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exp9_roc.png', dpi=100)
plt.show()

print("\nPlots saved: exp9_training_curves.png, exp9_confusion.png, exp9_roc.png")
