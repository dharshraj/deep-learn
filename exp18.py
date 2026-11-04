import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=" * 60)
print("  Experiment 18: CNN - Cats vs Dogs Classification")
print("=" * 60)

DATASET_DIR   = 'dogs-vs-cats'
TRAIN_DIR     = os.path.join(DATASET_DIR, 'train')
VAL_DIR       = os.path.join(DATASET_DIR, 'validation')
IMG_SIZE      = (150, 150)
BATCH_SIZE    = 32
EPOCHS        = 20

if not os.path.exists(TRAIN_DIR):
    print("\n[WARNING] Dataset folder not found at:", TRAIN_DIR)
    print("  Please download from https://www.kaggle.com/c/dogs-vs-cats/data")
    print("  and organize into train/cats, train/dogs, validation/cats, validation/dogs")
    print("\n  Running with a small synthetic demo instead...\n")

    import tensorflow as tf
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    mask_train = np.isin(y_train.flatten(), [3, 5])
    mask_test  = np.isin(y_test.flatten(), [3, 5])
    X_tr = X_train[mask_train].astype('float32') / 255.0
    y_tr = (y_train[mask_train].flatten() == 5).astype(int)
    X_te = X_test[mask_test].astype('float32')  / 255.0
    y_te = (y_test[mask_test].flatten() == 5).astype(int)

    print(f"Synthetic dataset — Train: {X_tr.shape}  |  Test: {X_te.shape}")
    print(f"Cats: {(y_tr==0).sum()}  Dogs: {(y_tr==1).sum()}")

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_tr, y_tr, epochs=20, batch_size=32,
                        validation_split=0.1, verbose=1)

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nTest Accuracy : {acc*100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    y_pred = (model.predict(X_te).flatten() >= 0.5).astype(int)
    correct = np.sum(y_pred == y_te)
    print(f"Correct Predictions: {correct}/{len(y_te)}")

else:
    print(f"Dataset found at: {DATASET_DIR}")

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary')
    val_data = val_gen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary')

    print(f"Train batches: {len(train_data)}  |  Val batches: {len(val_data)}")
    print(f"Classes: {train_data.class_indices}")

    model = Sequential([
        Conv2D(32,  (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64,  (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_data, epochs=EPOCHS,
                        validation_data=val_data, verbose=1)

    loss, acc = model.evaluate(val_data, verbose=0)
    print(f"\nValidation Accuracy : {acc*100:.2f}%")
    print(f"Validation Loss     : {loss:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history.history['accuracy'],     label='Train', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='orange')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Val',   color='orange')
axes[1].set_title('Binary Cross-Entropy Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Experiment 18 - CNN Cats vs Dogs', fontweight='bold')
plt.tight_layout()
plt.savefig('exp18_cats_dogs.png', dpi=100)
plt.show()

print("\nPlot saved: exp18_cats_dogs.png")
