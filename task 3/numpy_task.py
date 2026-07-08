import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Array:")
print(arr)

val1 = arr[0, 1]
val2 = arr[2, 2]
print("\nAccessed values:")
print("val1:", val1)
print("val2:", val2)

add_res = val1 + val2
sub_res = val2 - val1
mul_res = val1 * val2
div_res = val2 / val1
print("\nMath operations:")
print("Sum:", add_res)
print("Difference:", sub_res)
print("Product:", mul_res)
print("Division:", div_res)

seq = np.arange(10, 22)
print("\nArange:")
print(seq)

reshaped = np.reshape(seq, (3, 4))
print("\nReshaped:")
print(reshaped)

zeros = np.zeros((2, 2))
print("\nZeros:")
print(zeros)

ones = np.ones((2, 3))
print("\nOnes:")
print(ones)

print("\nSum of array:")
print(np.sum(arr))

print("\nMean of array:")
print(np.mean(arr))

print("\nStandard deviation of array:")
print(np.std(arr))

print("\nMin value:")
print(np.min(arr))

print("\nMax value:")
print(np.max(arr))

print("\nArgmin (index of min):")
print(np.argmin(arr))

print("\nArgmax (index of max):")
print(np.argmax(arr))

transposed = np.transpose(arr)
print("\nTransposed:")
print(transposed)

dot_res = np.dot(arr, transposed)
print("\nDot product with transpose:")
print(dot_res)

concat = np.concatenate((arr, arr), axis=0)
print("\nConcatenated:")
print(concat)

indices = np.where(arr > 5)
print("\nWhere arr > 5:")
print(indices)

unique_vals = np.unique(np.array([1, 1, 2, 2, 3]))
print("\nUnique:")
print(unique_vals)
