import numpy as np
import scipy as sp

# 题目 1: LU 分解 (基础)
# 给定矩阵 A

A = np.array([[4.0, 3, 1],
              [6, 3, 1],
              [8, 4, 1]])
print(A)

r = sp.linalg.lu(A)
# 置换矩阵
P = np.array(r[0])
# 下三角矩阵
L = np.array(r[1])
# 上三角矩阵
U = np.array(r[2])


# P @ A 不等于 L @ U
print(P @ A)
print(L @ U)

# 结果等于 A
print(P @ L @ U)

# 使用分解结果求解方程组 Ax = b，其中 b = [1, 0, 2]ᵀ

# P * L * U x = b => P * L * y = b

b = np.array([1, 0, 2]).T

pb = P.T @ b

y = sp.linalg.solve_triangular(L, pb, lower=True)

print(y)

x = sp.linalg.solve_triangular(U, y, lower=False)

print(x)
