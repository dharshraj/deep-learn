import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=" * 65)
print("  Experiment 22: Data Augmentation - Cats vs Dogs")
print("=" * 65)

DATASET_DIR = 'dogs-vs-cats'
TRAIN_DIR   = os.path.join(DATASET_DIR, 'train')
VAL_DIR     = os.path.join(DATASET_DIR, 'validation')
IMG_SIZE    = (150, 150)
BATCH       = 32
EPOCHS      = 20

dataset_available = os.path.exists(TRAIN_DIR)

if not dataset_available:
    print("\n[WARNING] dogs-vs-cats dataset not found.")
    print("  Using CIFAR-10 cats (class 3) vs dogs (class 5) as a substitute.\n")
    from tensorflow.keras.datasets import cifar10
    (X_tr_raw, y_tr_raw), (X_te_raw, y_te_raw) = cifar10.load_data()

    mask_tr = np.isin(y_tr_raw.flatten(), [3, 5])
    mask_te = np.isin(y_te_raw.flatten(), [3, 5])

    X_tr = X_tr_raw[mask_tr].astype('float32') / 255.0
    y_tr = (y_tr_raw[mask_tr].flatten() == 5).astype('float32')
    X_te = X_te_raw[mask_te].astype('float32') / 255.0
    y_te = (y_te_raw[mask_te].flatten() == 5).astype('float32')

    print(f"Train: {X_tr.shape}  |  Test: {X_te.shape}")
    print(f"Cats: {(y_tr==0).sum()}  Dogs: {(y_tr==1).sum()}")

    sample = X_tr[0:1]
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1
    )
    fig, axes = plt.subplots(2, 5, figsize=(13, 5))
    axes[0, 0].imshow(sample[0]); axes[0, 0].set_title('Original'); axes[0, 0].axis('off')
    for i, batch in enumerate(datagen.flow(sample, batch_size=1)):
        axes.flatten()[i+1].imshow(batch[0])
        axes.flatten()[i+1].set_title(f'Aug {i+1}')
        axes.flatten()[i+1].axis('off')
        if i >= 8: break
    plt.suptitle('Experiment 22 - Augmented Samples', fontweight='bold')
    plt.tight_layout(); plt.savefig('exp22_augmented_samples.png', dpi=100); plt.show()

    def build_model():
        m = Sequential([
            Conv2D(32,  (3,3), activation='relu', input_shape=(32,32,3)),
            MaxPooling2D(2,2),
            Conv2D(64,  (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return m

    print("\n  Training WITHOUT augmentation...")
    model_no_aug = build_model()
    hist_no_aug  = model_no_aug.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
                                     validation_split=0.1, verbose=1)

    print("\n  Training WITH augmentation...")
    aug_gen = datagen.flow(X_tr, y_tr, batch_size=BATCH)
    model_aug = build_model()
    hist_aug  = model_aug.fit(aug_gen, steps_per_epoch=len(X_tr)//BATCH,
                               epochs=EPOCHS,
                               validation_data=(X_te, y_te), verbose=1)

    loss_no, acc_no = model_no_aug.evaluate(X_te, y_te, verbose=0)
    loss_au, acc_au = model_aug.evaluate(X_te, y_te, verbose=0)

else:
    print(f"Dataset found at: {DATASET_DIR}")

    no_aug_gen = ImageDataGenerator(rescale=1./255)

    aug_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    def build_model():
        m = Sequential([
            Conv2D(32,  (3,3), activation='relu', input_shape=(150,150,3)),
            MaxPooling2D(2,2),
            Conv2D(64,  (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return m

    train_no_aug = no_aug_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                                   batch_size=BATCH, class_mode='binary')
    train_aug    = aug_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                               batch_size=BATCH, class_mode='binary')
    val_data     = val_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE,
                                               batch_size=BATCH, class_mode='binary')

    print("\n  Training WITHOUT augmentation...")
    model_no_aug = build_model()
    hist_no_aug  = model_no_aug.fit(train_no_aug, epochs=EPOCHS,
                                     validation_data=val_data, verbose=1)

    print("\n  Training WITH augmentation...")
    model_aug = build_model()
    hist_aug  = model_aug.fit(train_aug, epochs=EPOCHS,
                               validation_data=val_data, verbose=1)

    loss_no, acc_no = model_no_aug.evaluate(val_data, verbose=0)
    loss_au, acc_au = model_aug.evaluate(val_data, verbose=0)

print("\n" + "=" * 55)
print("         Data Augmentation Comparison")
print("=" * 55)
print(f"{'Model':<24} {'Test Acc (%)':>14} {'Test Loss':>12}")
print("-" * 52)
print(f"{'Without Augmentation':<24} {acc_no*100:>14.2f} {loss_no:>12.4f}")
print(f"{'With Augmentation':<24}    {acc_au*100:>14.2f} {loss_au:>12.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(hist_no_aug.history['accuracy'],     label='Train', color='royalblue')
axes[0,0].plot(hist_no_aug.history['val_accuracy'], label='Val',   color='orange')
axes[0,0].set_title('Without Augmentation - Accuracy'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(hist_no_aug.history['loss'],     label='Train', color='royalblue')
axes[0,1].plot(hist_no_aug.history['val_loss'], label='Val',   color='orange')
axes[0,1].set_title('Without Augmentation - Loss'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(hist_aug.history['accuracy'],     label='Train', color='forestgreen')
axes[1,0].plot(hist_aug.history['val_accuracy'], label='Val',   color='red')
axes[1,0].set_title('With Augmentation - Accuracy'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(hist_aug.history['loss'],     label='Train', color='forestgreen')
axes[1,1].plot(hist_aug.history['val_loss'], label='Val',   color='red')
axes[1,1].set_title('With Augmentation - Loss'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Experiment 22 - Data Augmentation Comparison', fontweight='bold')
plt.tight_layout(); plt.savefig('exp22_augmentation_comparison.png', dpi=100); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(hist_no_aug.history['val_accuracy'], label='Without Augmentation', color='royalblue',   linewidth=2)
plt.plot(hist_aug.history['val_accuracy'],    label='With Augmentation',    color='forestgreen', linewidth=2)
plt.title('Experiment 22 - Validation Accuracy Comparison')
plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('exp22_val_accuracy.png', dpi=100); plt.show()

print("\nPlots saved: exp22_augmented_samples.png, exp22_augmentation_comparison.png, exp22_val_accuracy.png")
