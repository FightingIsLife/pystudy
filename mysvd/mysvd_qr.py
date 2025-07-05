import numpy as np

# 作用：重建标准坐标系，减少计算的复杂度
A = np.array([[3, 4],
              [1, 3]])

# Q 是正交矩阵， R是上三角矩阵
Q, R = np.linalg.qr(A)

print("Q矩阵（旋转部分）:\n", Q)
print("Q T:\n", Q.T)
print("R矩阵（缩放部分）:\n", R)
print("重建A = Q@R:\n", Q @ R)