import numpy as np

# 创建一个一维数组，包含0到99的连续整数。
array99 = np.arange(0, 100, 1)
print(array99)

# 选取其中索引为0到20（包含20）的元素。
print(array99[0:21])

# 选取其中索引为20以后的元素。
print(array99[20:])

# 创建一个5x5的二维数组，元素为0到24（按行填充）。
array55 = array99[0:25].reshape(5, 5)
print(array55)

# 选取二维数组的第2行（行索引从0开始）。
print(array55[1])

# 选取二维数组的第3列。
print(array55[:, 2])

# 选取二维数组的一个子矩阵：行从1到3（不包括3），列从1到4（不包括4）。
print(array55[1:3, 1:4])

# 选取二维数组中满足条件大于10的所有元素。
# print(array55[np.where(array55 > 10)])
print(array55[array55 > 10])