#emp 20
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)

print("=" * 62)
print("  Experiment 20: CNN Performance Evaluation - CIFAR-10")
print("=" * 62)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

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
y_true = y_test.flatten()

precision = precision_score(y_true, y_pred, average='weighted')
recall    = recall_score(y_true, y_pred, average='weighted')
f1        = f1_score(y_true, y_pred, average='weighted')

print("\n" + "=" * 50)
print("         Overall Performance Metrics")
print("=" * 50)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")

print("\nPer-Class Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history.history['accuracy'],     label='Train', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='orange')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Val',   color='orange')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Experiment 20 - CNN Training Curves (CIFAR-10)', fontweight='bold')
plt.tight_layout()
plt.savefig('exp20_training_curves.png', dpi=100)
plt.show()

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set(xticks=np.arange(10), yticks=np.arange(10),
       xticklabels=class_names, yticklabels=class_names,
       title='Experiment 20 - Confusion Matrix (CIFAR-10)',
       ylabel='True Label', xlabel='Predicted Label')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(j, i, format(cm[i, j], 'd'),
            ha='center', va='center',
            color='white' if cm[i, j] > thresh else 'black', fontsize=8)
plt.tight_layout()
plt.savefig('exp20_confusion_matrix.png', dpi=100)
plt.show()

f1_per_class = f1_score(y_true, y_pred, average=None)
plt.figure(figsize=(10, 4))
bars = plt.bar(class_names, f1_per_class * 100, color='steelblue', edgecolor='black')
for bar, val in zip(bars, f1_per_class * 100):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', fontsize=9)
plt.title('Experiment 20 - Per-Class F1 Score')
plt.ylabel('F1 Score (%)'); plt.ylim(0, 100)
plt.xticks(rotation=30, ha='right'); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('exp20_f1_per_class.png', dpi=100)
plt.show()

print("\nPlots saved: exp20_training_curves.png, exp20_confusion_matrix.png, exp20_f1_per_class.png")

#emp 21
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

#emp 22
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

#emp 23
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("=" * 65)
print("  Experiment 23: Transfer Learning - EfficientNetB0 / ResNet50")
print("=" * 65)

DATASET_BASE = 'intel-image-classification'
TRAIN_DIR    = os.path.join(DATASET_BASE, 'seg_train', 'seg_train')
TEST_DIR     = os.path.join(DATASET_BASE, 'seg_test',  'seg_test')

IMG_SIZE    = (224, 224)
BATCH       = 32
EPOCHS      = 15
NUM_CLASSES = 6
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

dataset_available = os.path.exists(TRAIN_DIR)

if not dataset_available:
    print("\n[WARNING] Intel Image Classification dataset not found.")
    print("  Using CIFAR-10 resized to 224x224 as a placeholder (6 classes).\n")

    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical

    (X_tr, y_tr), (X_te, y_te) = cifar10.load_data()
    mask_tr = (y_tr.flatten() < 6)
    mask_te = (y_te.flatten() < 6)
    X_tr = X_tr[mask_tr].astype('float32') / 255.0
    y_tr = y_tr[mask_tr].flatten()
    X_te = X_te[mask_te].astype('float32') / 255.0
    y_te = y_te[mask_te].flatten()

    X_tr_r = tf.image.resize(X_tr, (224, 224)).numpy()
    X_te_r = tf.image.resize(X_te, (224, 224)).numpy()

    y_tr_cat = to_categorical(y_tr, 6)
    y_te_cat = to_categorical(y_te, 6)

    print(f"Train: {X_tr_r.shape}  |  Test: {X_te_r.shape}")

    def build_transfer_model(base_model_fn, trainable_base=False):
        base = base_model_fn(weights='imagenet', include_top=False,
                             input_shape=(224, 224, 3))
        base.trainable = trainable_base
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        output = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    print("\n  Phase 1: Feature Extraction (frozen base) - EfficientNetB0")
    model_eff = build_transfer_model(EfficientNetB0, trainable_base=False)
    model_eff.summary()

    hist_eff = model_eff.fit(X_tr_r, y_tr_cat, epochs=EPOCHS, batch_size=BATCH,
                              validation_split=0.1, callbacks=callbacks, verbose=1)

    loss_eff, acc_eff = model_eff.evaluate(X_te_r, y_te_cat, verbose=0)
    print(f"\nEfficientNetB0 Test Accuracy : {acc_eff*100:.2f}%")

    print("\n  Phase 1: Feature Extraction (frozen base) - ResNet50")
    model_res = build_transfer_model(ResNet50, trainable_base=False)
    hist_res  = model_res.fit(X_tr_r, y_tr_cat, epochs=EPOCHS, batch_size=BATCH,
                               validation_split=0.1, callbacks=callbacks, verbose=1)

    loss_res, acc_res = model_res.evaluate(X_te_r, y_te_cat, verbose=0)
    print(f"\nResNet50 Test Accuracy : {acc_res*100:.2f}%")

    print("\n  Phase 2: Fine-Tuning EfficientNetB0 (top layers unfrozen)...")
    base = model_eff.layers[0]
    for layer in base.layers[-20:]:
        layer.trainable = True

    model_eff.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    hist_ft = model_eff.fit(X_tr_r, y_tr_cat, epochs=10, batch_size=BATCH,
                             validation_split=0.1, callbacks=callbacks, verbose=1)

    loss_ft, acc_ft = model_eff.evaluate(X_te_r, y_te_cat, verbose=0)
    print(f"\nEfficientNetB0 Fine-tuned Test Accuracy : {acc_ft*100:.2f}%")

    y_pred = np.argmax(model_eff.predict(X_te_r), axis=1)

else:
    print(f"Dataset found at: {DATASET_BASE}")

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
        validation_split=0.1
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='categorical', subset='training')
    val_data   = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='categorical', subset='validation')
    test_data  = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='categorical', shuffle=False)

    class_names = list(train_data.class_indices.keys())
    NUM_CLASSES = len(class_names)
    print(f"Classes: {class_names}")

    def build_transfer_model(base_model_fn, trainable_base=False):
        base = base_model_fn(weights='imagenet', include_top=False,
                             input_shape=(224, 224, 3))
        base.trainable = trainable_base
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        output = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    print("\n  Phase 1: Feature Extraction - EfficientNetB0 (frozen base)")
    model_eff = build_transfer_model(EfficientNetB0, trainable_base=False)
    model_eff.summary()
    hist_eff  = model_eff.fit(train_data, epochs=EPOCHS, validation_data=val_data,
                               callbacks=callbacks, verbose=1)

    loss_eff, acc_eff = model_eff.evaluate(test_data, verbose=0)
    print(f"\nEfficientNetB0 Test Accuracy: {acc_eff*100:.2f}%")

    print("\n  Phase 2: Fine-tuning - EfficientNetB0 (last 20 layers)")
    for layer in model_eff.layers[0].layers[-20:]:
        layer.trainable = True
    model_eff.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    hist_ft = model_eff.fit(train_data, epochs=10, validation_data=val_data,
                             callbacks=callbacks, verbose=1)
    loss_ft, acc_ft = model_eff.evaluate(test_data, verbose=0)
    print(f"\nEfficientNetB0 Fine-tuned Test Accuracy: {acc_ft*100:.2f}%")

    y_pred   = np.argmax(model_eff.predict(test_data), axis=1)
    y_te     = test_data.classes
    loss_res, acc_res = loss_eff, acc_eff

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(hist_eff.history['accuracy'],     label='Train', color='royalblue')
axes[0].plot(hist_eff.history['val_accuracy'], label='Val',   color='orange')
axes[0].set_title('EfficientNetB0 - Feature Extraction Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_eff.history['loss'],     label='Train', color='royalblue')
axes[1].plot(hist_eff.history['val_loss'], label='Val',   color='orange')
axes[1].set_title('EfficientNetB0 - Feature Extraction Loss')
axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Experiment 23 - Transfer Learning (EfficientNetB0)', fontweight='bold')
plt.tight_layout()
plt.savefig('exp23_transfer_learning.png', dpi=100)
plt.show()

print("\n" + "=" * 60)
print("           Transfer Learning Summary")
print("=" * 60)
print(f"  EfficientNetB0 (Feature Extraction) : {acc_eff*100:.2f}%")
print(f"  ResNet50       (Feature Extraction) : {acc_res*100:.2f}%")
print(f"  EfficientNetB0 (After Fine-Tuning)  : {acc_ft*100:.2f}%")
print("\nFrozen base layers  -> only classification head trained")
print("Fine-tuning         -> last 20 layers unfrozen, very low lr (1e-5)")

try:
    print("\nClassification Report (EfficientNetB0 Fine-tuned):")
    print(classification_report(y_te, y_pred, target_names=class_names))
except Exception:
    pass

try:
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Experiment 23 - Confusion Matrix (EfficientNetB0)')
    plt.savefig('exp23_confusion.png', dpi=100)
    plt.show()
except Exception:
    pass

print("\nPlots saved: exp23_transfer_learning.png, exp23_confusion.png")
