import numpy as np
marks = np.array([85, 92, 76, 98, 88])
print("Marks:", marks)
print("Mean:", np.mean(marks))
print("Minimum:", np.min(marks))
print("Maximum:", np.max(marks))
print("Standard Deviation:", np.std(marks))
topper = np.argmax(marks)
print("Topper Position (index):", topper)
print("Topper Marks:", marks[topper])
max_marks = marks[0]
pos = 0
for i in range(1, len(marks)):
    if marks[i] > max_marks:
        max_marks = marks[i]
        pos = i
print("Manual Topper Position:", pos)
print("Check:", topper == pos)
