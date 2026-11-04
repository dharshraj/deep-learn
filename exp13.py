import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

print("=" * 60)
print("  Experiment 13: L2 Regularization - Breast Cancer Dataset")
print("=" * 60)

data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Features: {X.shape[1]}  |  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

def build_no_l2():
    model = Sequential([
        Dense(128, input_shape=(30,), activation='relu'),
        Dense(64,  activation='relu'),
        Dense(32,  activation='relu'),
        Dense(1,   activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_with_l2(l2_lambda=0.01):
    reg = regularizers.l2(l2_lambda)
    model = Sequential([
        Dense(128, input_shape=(30,), activation='relu', kernel_regularizer=reg),
        Dense(64,  activation='relu', kernel_regularizer=reg),
        Dense(32,  activation='relu', kernel_regularizer=reg),
        Dense(1,   activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

EPOCHS = 100
BATCH  = 32

print("\n  Training WITHOUT L2 regularization...")
model_no_l2 = build_no_l2()
hist_no_l2  = model_no_l2.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                                validation_split=0.1, verbose=0)

print("  Training WITH L2 regularization (λ=0.01)...")
model_l2 = build_with_l2(0.01)
hist_l2  = model_l2.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                          validation_split=0.1, verbose=0)

l2_values   = [0.001, 0.01, 0.1]
l2_results  = {}
for lam in l2_values:
    m = build_with_l2(lam)
    h = m.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
              validation_split=0.1, verbose=0)
    loss, acc = m.evaluate(X_test, y_test, verbose=0)
    l2_results[lam] = {'acc': acc, 'loss': loss, 'hist': h}

loss_nd, acc_nd = model_no_l2.evaluate(X_test, y_test, verbose=0)
loss_l2, acc_l2 = model_l2.evaluate(X_test,   y_test, verbose=0)

print("\n" + "=" * 55)
print("               Results Summary")
print("=" * 55)
print(f"{'Model':<28} {'Test Acc (%)':>14} {'Test Loss':>10}")
print("-" * 55)
print(f"{'Without L2':<28} {acc_nd*100:>14.2f} {loss_nd:>10.4f}")
print(f"{'With L2 (λ=0.01)':<28} {acc_l2*100:>14.2f} {loss_l2:>10.4f}")
for lam in l2_values:
    print(f"{'With L2 (λ='+str(lam)+')':<28} {l2_results[lam]['acc']*100:>14.2f} {l2_results[lam]['loss']:>10.4f}")

gap_nd = max(hist_no_l2.history['accuracy']) - max(hist_no_l2.history['val_accuracy'])
gap_l2 = max(hist_l2.history['accuracy'])    - max(hist_l2.history['val_accuracy'])
print(f"\nOverfitting Gap:")
print(f"  Without L2 : {gap_nd:.4f}")
print(f"  With L2    : {gap_l2:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(hist_no_l2.history['accuracy'],     label='Train', color='royalblue')
axes[0,0].plot(hist_no_l2.history['val_accuracy'], label='Val',   color='orange')
axes[0,0].set_title('Without L2 - Accuracy')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(hist_no_l2.history['loss'],     label='Train', color='royalblue')
axes[0,1].plot(hist_no_l2.history['val_loss'], label='Val',   color='orange')
axes[0,1].set_title('Without L2 - Loss')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(hist_l2.history['accuracy'],     label='Train', color='forestgreen')
axes[1,0].plot(hist_l2.history['val_accuracy'], label='Val',   color='red')
axes[1,0].set_title('With L2 (λ=0.01) - Accuracy')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(hist_l2.history['loss'],     label='Train', color='forestgreen')
axes[1,1].plot(hist_l2.history['val_loss'], label='Val',   color='red')
axes[1,1].set_title('With L2 (λ=0.01) - Loss')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Experiment 13 - L2 Regularization on Breast Cancer Dataset', fontweight='bold')
plt.tight_layout()
plt.savefig('exp13_l2_comparison.png', dpi=100)
plt.show()

plt.figure(figsize=(7, 4))
lambdas = [0] + l2_values
accs    = [acc_nd*100] + [l2_results[l]['acc']*100 for l in l2_values]
plt.plot(['No L2']+[str(l) for l in l2_values], accs,
         marker='o', color='royalblue', linewidth=2, markersize=8)
plt.title('Experiment 13 - Test Accuracy vs L2 Lambda')
plt.xlabel('L2 Lambda'); plt.ylabel('Test Accuracy (%)')
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('exp13_lambda_effect.png', dpi=100)
plt.show()

print("\nPlots saved: exp13_l2_comparison.png, exp13_lambda_effect.png")
