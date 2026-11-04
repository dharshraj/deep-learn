import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from scipy.ndimage import convolve

print("=" * 60)
print("  Experiment 16: Convolution and Pooling on CIFAR-10")
print("=" * 60)

(X_train, y_train), _ = cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

sample_img  = X_train[0]
gray_img    = np.mean(sample_img, axis=2)
gray_norm   = gray_img / 255.0
label_name  = class_names[y_train[0][0]]

print(f"Sample image label : {label_name}")
print(f"Image shape        : {sample_img.shape}")

kernel_edge_h = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=float)

kernel_edge_v = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=float)

kernel_sharp = np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]], dtype=float)

kernel_blur = np.ones((3, 3), dtype=float) / 9.0

edge_h  = convolve(gray_norm, kernel_edge_h)
edge_v  = convolve(gray_norm, kernel_edge_v)
sharp   = convolve(gray_norm, kernel_sharp)
blurred = convolve(gray_norm, kernel_blur)

def max_pool2d(img, pool_size=2):
    h, w = img.shape
    out  = np.zeros((h // pool_size, w // pool_size))
    for i in range(0, h - pool_size + 1, pool_size):
        for j in range(0, w - pool_size + 1, pool_size):
            out[i//pool_size, j//pool_size] = np.max(img[i:i+pool_size, j:j+pool_size])
    return out

def avg_pool2d(img, pool_size=2):
    h, w = img.shape
    out  = np.zeros((h // pool_size, w // pool_size))
    for i in range(0, h - pool_size + 1, pool_size):
        for j in range(0, w - pool_size + 1, pool_size):
            out[i//pool_size, j//pool_size] = np.mean(img[i:i+pool_size, j:j+pool_size])
    return out

max_pooled = max_pool2d(gray_norm, 2)
avg_pooled = avg_pool2d(gray_norm, 2)

print(f"\nOriginal image size    : {gray_norm.shape}")
print(f"After Max Pooling (2x2): {max_pooled.shape}")
print(f"After Avg Pooling (2x2): {avg_pooled.shape}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'Experiment 16 - Convolution & Pooling  (Image: {label_name})',
             fontsize=13, fontweight='bold')

images = [
    (sample_img,  'Original (RGB)',          None),
    (gray_norm,   'Grayscale',               'gray'),
    (edge_h,      'Horizontal Edge (Sobel)', 'gray'),
    (edge_v,      'Vertical Edge (Sobel)',   'gray'),
    (sharp,       'Sharpened',               'gray'),
    (blurred,     'Blurred (3×3 Avg)',       'gray'),
    (max_pooled,  'Max Pooling (2×2)',        'gray'),
    (avg_pooled,  'Average Pooling (2×2)',    'gray'),
]

for ax, (img, title, cmap) in zip(axes.flatten(), images):
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('exp16_convolution_pooling.png', dpi=100)
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
filters = [(kernel_edge_h, 'Sobel H'), (kernel_edge_v, 'Sobel V'),
           (kernel_sharp, 'Sharpen'), (kernel_blur, 'Blur')]
for ax, (k, title) in zip(axes, filters):
    im = ax.imshow(k, cmap='RdBu', interpolation='nearest')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
plt.suptitle('Experiment 16 - Filter Kernels', fontweight='bold')
plt.tight_layout()
plt.savefig('exp16_kernels.png', dpi=100)
plt.show()

print("\nPlots saved: exp16_convolution_pooling.png, exp16_kernels.png")
