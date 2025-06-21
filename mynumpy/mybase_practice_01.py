import numpy as np

# 4.1 创建两个矩阵：
#
# A = [[1, 2], [3, 4]]
# B = [[5, 6], [7, 8]]

A = np.arange(1, 5).reshape(2, 2)
B = np.arange(5, 9).reshape(2, 2)
print(A)
print(B)

# 计算A与B的矩阵乘法。
multiAB = A @ B
print(multiAB)

# 计算矩阵A的逆矩阵（如果不可逆，则尝试伪逆）。
det = np.linalg.det(A)
print(det)
if det == 0:
    print(np.linalg.pinv(A))
else:
    print(np.linalg.inv(A))

# 求解线性方程组：Ax = b，其中b=[5, 11]（即方程组：x+2y=5, 3x+4y=11）。
# Ax=b 等价于 A⁻¹Ax = A⁻¹b  等价于 x = A⁻¹b
print("\n")
b = np.array([5, 11]).T
result = np.linalg.inv(A) @ b
print(result)
