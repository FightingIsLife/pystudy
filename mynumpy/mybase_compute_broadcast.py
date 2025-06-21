import numpy as np

# 创建两个3x3的数组，元素为1到9（按行）和9到1（按行递减），进行逐元素相加、相减、相乘、相除、整除、取模
arr1 = np.arange(1, 10).reshape(3, 3)
arr2 = np.arange(9, 0, -1).reshape(3, 3)
print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 * arr2)
print(arr1 / arr2)
print(arr1 // arr2)
print(arr1 % arr2)

# 对数组的每个元素求平方（使用两种方法：乘运算符和np.square）。
print(arr1 * arr1)
print(np.square(arr1))

# 对数组的每个元素求正弦（np.sin）和指数（np.exp）。
print(np.sin(arr1))
print(np.exp(arr1))

# 创建一个3x3的数组和一个1x3的数组（一维数组也可），进行逐元素相加（注意广播）。
arr33 = np.ones((3, 3))
arr3 = np.arange(3)
print("=== 行方向广播 ===")
print(arr33 + arr3)

print("\n=== 列方向广播 ===")
print(arr33 + arr3[:, np.newaxis])

# 创建一个3x1的数组和一个1x3的数组，进行相加，观察结果。
arr31 = np.ones((3, 1))
arr13 = np.ones((1, 3))
print(arr31 + arr13)

# 创建一个3x3的数组和一个标量（如10），进行相加和相乘。
print(arr33 + 10)
print(arr33 * 10)

# 创建两个2x2的矩阵，进行矩阵乘法（使用np.dot和@运算符）。
arr22_1 = np.arange(1, 5).reshape(2, 2)
arr22_2 = np.arange(1, 5).reshape(2, 2)

print(arr22_1)
print(arr22_2)
print(np.dot(arr22_1, arr22_2))
print(arr22_1 @ arr22_2)

# 创建一个3x2的矩阵和一个2x4的矩阵，进行矩阵乘法。
arr32 = np.arange(0, 6).reshape(3, 2)
arr24 = np.arange(0, 8).reshape(2, 4)
print(arr32)
print(arr24)
print(np.dot(arr32, arr24))
print(arr32 @ arr24)