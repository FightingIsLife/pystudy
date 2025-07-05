import numpy as np

# 创建一个秩为2的矩阵 (3x2)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# 执行 SVD，U是正交矩阵，V也是正交矩阵
U, s, Vt = np.linalg.svd(A)  # 注意！s 是一个一维数组，包含奇异值 (σ₁, σ₂)

print("原始矩阵 A:")
print(A)

print("\n左奇异向量矩阵 U:")
print(U)  # 形状 (3, 3)

print("\n奇异值 s (σ):")  # NumPy 返回奇异值数组，不是矩阵！
print(s)  # [9.52551809, 0.51430058] 注意只有两个奇异值，因为 rank=2

# 手动构造对角矩阵 Sigma (3x2)。Σ[i, i] = s[i], 其它为0
Sigma = np.zeros(A.shape)  # 创建一个和A一样大(3x2)的零矩阵
k = len(s)  # 非零奇异值的个数 (2)
for i in range(k):
    Sigma[i, i] = s[i]  # 把s[i]放到对角线上

print("\n奇异值矩阵 Σ (填充后):")
print(Sigma)

print("\n右奇异向量矩阵的转置 Vᵀ:")
print(Vt)  # 形状 (2, 2)

# 重构原始矩阵 A
A_reconstructed = U @ Sigma @ Vt  # U (3x3) * Σ (3x2) * Vt (2x2) = (3x2)

print("\n重构矩阵 (U * Σ * Vᵀ):")
print(A_reconstructed)  # 应该非常接近原始 A