import numpy as np

# 创建两个三维数组
A = np.ones((2, 2, 2))  # 形状 (2,2,2)
B = np.ones((2, 2, 2))  # 形状 (2,2,2)

print("数组 A:\n", A)
print("\n数组 B:\n", B)

# 使用 @/matmul
matmul_result = A @ B
print("\n@/matmul 结果 (形状 {}):\n{}".format(matmul_result.shape, matmul_result))

# 使用 dot
dot_result = np.dot(A, B)
print("\ndot 结果 (形状 {}):\n{}".format(dot_result.shape, dot_result))


A = np.array([[4, 3, 1],
              [6, 3, 1],
              [8, 4, 1]])

b = np.array([1, 0, 2])


print(b)
print(A @ b)
print(b.shape)
print((A @ b).shape)

t = np.array([1, 0, 2]) @ A
print(t)
print(t.shape)

