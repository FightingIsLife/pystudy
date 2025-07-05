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

# ================== 使用 SciPy 的 QR 分解求解 ==================
# 1. 计算 QR 分解
Q, R = la.qr(X, mode='economic')  # 'economic'模式返回精简QR分解

# 2. 计算最小二乘解
# 公式: Rβ = Qᵀy
beta = la.solve_triangular(R, Q.T @ y, lower=False)

print("SciPy QR分解计算的最小二乘解:")
print(f"斜率 a = {beta[0]:.6f}")
print(f"截距 b = {beta[1]:.6f}")

# ================== 使用 SciPy 的专用最小二乘函数 ==================
# 更简单直接的方法
beta_lstsq, residuals, rank, s = la.lstsq(X, y)

print("\nSciPy lstsq计算的最小二乘解:")
print(f"斜率 a = {beta_lstsq[0]:.6f}")
print(f"截距 b = {beta_lstsq[1]:.6f}")

# ================== 模型预测和可视化 ==================
# 生成预测值
x_points = np.linspace(0, 6, 100)
y_pred = beta[0] * x_points + beta[1]

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, color='red', s=80, label='Original data points')
plt.plot(x_points, y_pred, 'b-', linewidth=2, label=f'y = {beta[0]:.2f}x + {beta[1]:.2f}')
plt.title('Linear Regression: Least Squares Solution', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))
plt.tight_layout()

# 添加残差线
for i in range(len(X)):
    y_pred_i = beta[0] * X[i, 0] + beta[1]
    plt.plot([X[i, 0], X[i, 0]], [y[i], y_pred_i], 'r--', alpha=0.7)

plt.show()

# ================== 残差分析 ==================
y_pred_all = X @ beta
residuals = y - y_pred_all

print("\n残差分析:")
print(f"残差平方和 (SSE): {np.sum(residuals**2):.6f}")
print(f"最大残差: {np.max(np.abs(residuals)):.6f}")
print(f"平均绝对残差: {np.mean(np.abs(residuals)):.6f}")

# 残差图
plt.figure(figsize=(10, 4))
plt.scatter(y_pred_all, residuals, color='green', s=80)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
plt.title('Residual Plot', fontsize=14)
plt.xlabel('Predicted values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()