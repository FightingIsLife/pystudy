import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# 线性模型为 y = a*x + b，转化为矩阵方程 Xβ = y
# X * β = [1   1]   *   [a]   =   [1*a + 1*b]   =   [a*1 + b*1]
#         [2   1]       [b]       [2*a + 1*b]       [a*2 + b*1]
#         [3   1]                  [3*a + 1*b]       [a*3 + b*1]

# 数据准备
X = np.array([[1, 1], [2, 1], [3, 1], [3.4, 1], [5.5, 1]], dtype=float)
y = np.array([2, 3, 4, 5, 6], dtype=float)

# ================== 使用 SVD 分解求解 ==================
# 1. 计算 svd 分解
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# 2. 计算最小二乘解
# 公式: β = V * Σ⁺ * Uᵀ * y
# 创建一个与s相同形状的数组，存放伪逆值
s_inv = np.zeros_like(s)
threshold = 1e-10  # 奇异值阈值

# 只对大于阈值的奇异值取倒数 (避免除以0)
s_inv[s > threshold] = 1 / s[s > threshold]

# 构造伪逆Σ⁺（按正确维度）
# 注意：在简化SVD中，Σ⁺是2×2对角矩阵
Sigma_plus  = np.diag(s_inv)

beta = Vt.T @ Sigma_plus  @ U.T @ y

print("svd 分解计算的最小二乘解:")
print(f"斜率 a = {beta[0]:.6f}")
print(f"截距 b = {beta[1]:.6f}")

