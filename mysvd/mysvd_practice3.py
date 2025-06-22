import numpy as np
import timeit

# 生成一个 100×100 的随机矩阵 F
F = np.random.rand(100, 100)

# 计算 F 的 QR 分解
def qr_decomp(F):
    Q, R = np.linalg.qr(F)
    return Q, R

# 计算 F 的 SVD 分解
def svd_decomp(F):
    U, s, V = np.linalg.svd(F, full_matrices=False)
    return U, s, V

# 比较两种分解的计算时间
print("\n===== 性能测试 =====")

# 使用 timeit 模块替代 %timeit
n_repeats = 100
qr_time = timeit.timeit(lambda: qr_decomp(F), number=n_repeats)
svd_time = timeit.timeit(lambda: svd_decomp(F), number=n_repeats)

print(f"QR 分解平均耗时: {qr_time/n_repeats:.6f} 秒 (n={n_repeats})")
print(f"SVD 分解平均耗时: {svd_time/n_repeats:.6f} 秒 (n={n_repeats})")

# 使用 SVD 计算条件数
def compute_condition_number(F):
    _, s, _ = np.linalg.svd(F)
    cond = s[0] / s[-1]
    return cond

condition_num = compute_condition_number(F)
print(f"\n使用 SVD 计算的条件数: {condition_num:.4f}")