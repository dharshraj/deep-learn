import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("=" * 58)
print("  Experiment 8: MSE Loss - California Housing Regression")
print("=" * 58)

housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"Dataset Shape : {X.shape}")
print(f"Features      : {housing.feature_names}")
print(f"Target Range  : {y.min():.2f} to {y.max():.2f} (in $100,000s)")

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTrain samples : {X_train.shape[0]}")
print(f"Test samples  : {X_test.shape[0]}")

model = Sequential([
    Dense(128, input_shape=(8,), activation='relu'),
    Dense(64,  activation='relu'),
    Dense(32,  activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=64,
                    validation_split=0.1, verbose=1)

mse, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MSE : {mse:.4f}")
print(f"Test MAE : {mae:.4f}")
print(f"Test RMSE: {np.sqrt(mse):.4f}")

y_pred = model.predict(X_test).flatten()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(history.history['loss'],     label='Train Loss', color='royalblue')
axes[0].plot(history.history['val_loss'], label='Val Loss',   color='orange')
axes[0].set_title('MSE Loss During Training')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'],     label='Train MAE', color='forestgreen')
axes[1].plot(history.history['val_mae'], label='Val MAE',   color='red')
axes[1].set_title('MAE During Training')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Experiment 8 - California Housing Regression', fontweight='bold')
plt.tight_layout()
plt.savefig('exp8_training_curves.png', dpi=100)
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='royalblue', s=8)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Fit')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Price ($100k)')
plt.ylabel('Predicted Price ($100k)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exp8_actual_vs_predicted.png', dpi=100)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(9, 4))
plt.scatter(y_pred, residuals, alpha=0.3, color='purple', s=8)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Plot')
plt.xlabel('Predicted Price ($100k)')
plt.ylabel('Residual')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exp8_residuals.png', dpi=100)
plt.show()

print("\nPlots saved: exp8_training_curves.png, exp8_actual_vs_predicted.png, exp8_residuals.png")
