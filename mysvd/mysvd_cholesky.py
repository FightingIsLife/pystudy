import numpy as np
import scipy as sp

# 作用：针对正定矩阵计算和存储的优化，正定矩阵使用cholesky相比lu性能和存储能够优化1倍
A = np.array([[4, 2],
[2, 5]])
C = sp.linalg.cholesky(A)
print(C)
print(C.T)
# 正定矩阵
print(C.T @ C)