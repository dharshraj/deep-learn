import numpy as np

roll = np.arange(1, 61)
time = np.linspace(0, 1, 50)
zero = np.zeros(8)
one = np.ones(8)

print("Roll Numbers:", roll)
print("Time Points:", time)
print("Zeros:", zero)
print("Ones:", one)

print("Roll - Length:", len(roll), "Dtype:", roll.dtype)
print("Time - Length:", len(time), "Dtype:", time.dtype)
print("Zeros - Length:", len(zero), "Dtype:", zero.dtype)
print("Ones - Length:", len(one), "Dtype:", one.dtype)
