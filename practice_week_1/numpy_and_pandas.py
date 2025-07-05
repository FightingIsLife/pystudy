import pandas as pd
import numpy as np

# 假设我们在题目1中得到一个非常大的RFM得分矩阵（客户数很多，比如100万客户），每个客户有三个得分（R, F, M）和客户ID。为了便于传输，我们想对得分矩阵进行压缩。
# 生成模拟RFM数据
np.random.seed(42)
size = 10000  # 1000个客户
rfm_scores = pd.DataFrame({
    'R_score': np.random.randint(1, 5, size),
    'F_score': np.random.randint(1, 5, size),
    'M_score': np.random.randint(1, 5, size),
    'N_score': np.random.randint(1, 5, size),
    'O_score': np.random.randint(1, 5, size),
    'P_score': np.random.randint(1, 5, size),
    'Q_score': np.random.randint(1, 5, size),
})

# 请实现：
# 1. 用Pandas计算客户总分：总分 = R*100 + F*10 + M (RFM模型经典编码)
rfm_scores['total_score'] = rfm_scores.R_score * 1000000 + rfm_scores.F_score * 100000 + rfm_scores.M_score * 10000 + rfm_scores.N_score * 1000 + rfm_scores.O_score * 100 + rfm_scores.P_score * 10 + rfm_scores.Q_score

print(rfm_scores.shape)

# 2. 将RFM得分矩阵(rfm_scores)中心化,用NumPy进行SVD分解

centered_cols = ['R_score', 'F_score', 'M_score', 'N_score', 'O_score', 'P_score', 'Q_score']
rfm_centered = rfm_scores[centered_cols] - rfm_scores[centered_cols].mean()

U, s, Vt = np.linalg.svd(rfm_centered, full_matrices=False)

print(U.shape)
print(Vt.shape)

# 3. 保留90%能量（奇异值平方和占比）所需的最小k值
s_squared = s ** 2
total_energy = np.sum(s_squared)
print(f"总能量: {total_energy:.4f}")
print(f"奇异值: {s}")

cumulative_ratio = np.cumsum(s_squared) / total_energy
print("累积能量占比:", cumulative_ratio)

k = np.argmax(cumulative_ratio >= 0.90) + 1
print(f"保留90%能量所需的最小k值: {k}")

# 4. 用前k个奇异值重建矩阵，并反中心化
k=3
S_matrix = np.zeros((U.shape[1], Vt.shape[0]))
S_matrix[:k, :k] = np.diag(s[:k])
reconstructed = U @ S_matrix @ Vt + rfm_scores[centered_cols].mean().values

# 转换为DataFrame
result = pd.DataFrame(reconstructed, columns=centered_cols)
result['rebuilt_total'] = result.R_score * 1000000 + result.F_score * 100000 + result.M_score * 10000 + result.N_score * 1000 + result.O_score * 100 + result.P_score * 10 + result.Q_score

# 5. 计算相关系数 (补充)
from scipy.stats import pearsonr
corr, _ = pearsonr(rfm_scores.total_score, result.rebuilt_total)
print(f"总分相关系数: {corr:.4f}")

# 可选：验证重建质量
relative_error = np.linalg.norm(reconstructed - rfm_scores[centered_cols], 'fro') / np.linalg.norm(rfm_scores[centered_cols], 'fro')
print(f"相对重建误差: {relative_error:.4f}")
