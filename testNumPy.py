import numpy as np

# Java数组 vs NumPy数组
java_array = [1, 2, 3, 4, 5]  # 传统列表
np_array = np.array([1, 2, 3, 4, 5])  # NumPy数组

print("Java数组操作：", [x * 2 for x in java_array])  # 需要循环
print("NumPy向量化：", np_array * 2)  # 整个数组操作