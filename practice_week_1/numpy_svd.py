import numpy as np

matrix = np.array([
    [1, 0, 2, 4],
    [3, 4, 1, 2],
    [0, 5, 3, 1]
], dtype=np.float32)

# 1. 对matrix进行SVD分解
U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

# 2. 保留前2个奇异值（其他置零），重建矩阵
k = 2

# 创建奇异值对角矩阵（仅保留前k个）
s_k = np.diag(s[:k])

# 取U和Vt的前k列/行
U_k = U[:, :k]
Vt_k = Vt[:k, :]  # 更明确的行切片

# 重建矩阵
R = U_k @ s_k @ Vt_k

# 3. 计算重建矩阵与原矩阵的Frobenius范数误差
frobenius_error = np.linalg.norm(matrix - R, ord='fro')

# 4. 打印保留的奇异值数量及误差
print(f"保留的奇异值数量: {k}")
print(f"重建误差 (Frobenius范数): {frobenius_error:.6f}")
print("\n重建矩阵:")
print(R)