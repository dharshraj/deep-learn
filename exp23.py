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
