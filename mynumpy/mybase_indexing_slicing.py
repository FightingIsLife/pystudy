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


# 垂直与水平堆叠
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

v_stack = np.vstack((a, b))        # 垂直堆叠
h_stack = np.hstack((a, b.T))       # 水平堆叠
print(f"垂直堆叠:\n{v_stack}\n水平堆叠:\n{h_stack}")

## 通用concatenate方法
c = np.array([7, 8])
concat_axis0 = np.concatenate((a, [c]), axis=0)  # 沿0轴(行)
concat_axis1 = np.concatenate((a, [[9], [10]]), axis=1)  # 沿1轴(列)
print(f"沿0轴连接:\n{concat_axis0}\n沿1轴连接:\n{concat_axis1}")

# 行列分割
arr = np.arange(12).reshape(3, 4)
split_arr = np.split(arr, [1, 3], axis=1)  # 在第1列和第3列后切割
print(f"原始数组:\n{arr}\n分割结果:")
for part in split_arr:
    print(part)

vsplit_arr = np.vsplit(arr, 3)      # 垂直分割
hsplit_arr = np.hsplit(arr, 2)      # 水平分割
print(f"垂直分割:\n{vsplit_arr[0]}\n水平分割:\n{hsplit_arr[0]}")
