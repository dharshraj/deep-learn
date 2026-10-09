import numpy as np

steps = np.array([
    5000, 6000, 7500, 8000, 6500,
    9000, 10000, 7000, 8500, 9500,
    11000, 8000, 7500, 9000, 10500,
    6000, 7000, 8500, 9500, 10000,
    8000, 9000, 11000, 12000, 7500,
    8500, 9500, 10500, 11500, 12500
])

first_week = steps[:7]
last_five = steps[-5:]
alternate = steps[::2]
reverse = steps[::-1]

print("First Week:", first_week)
print("Last 5 Days:", last_five)
print("Every Alternate Day:", alternate)
print("Reversed:", reverse)

print("Lengths:",
      len(first_week),
      len(last_five),
      len(alternate),
      len(reverse))
