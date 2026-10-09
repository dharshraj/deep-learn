import numpy as np

week1 = np.array([100, 150, 200, 120, 180, 90, 160])
week2 = np.array([120, 140, 190, 150, 170, 110, 155])

total = week1 + week2
change = week2 - week1

print("Week 1 Sales:", week1)
print("Week 2 Sales:", week2)
print("Total Sales:", total)
print("Change in Sales:", change)

fallen = np.where(change < 0)[0]
print("Products with Fallen Sales (indices):", fallen)
