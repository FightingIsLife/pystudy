import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. 加载图片 - 我们选择一张简单的黑白图片
image_path = 'C:\\Users\\Administrator\\Pictures\\color\\out26.png'  # 替换为你自己的图片路径
original_image = Image.open(image_path).convert('L')  # 转换为灰度图
image_matrix = np.array(original_image)  # 将图像转换为数值矩阵

print(f"原始图像尺寸: {image_matrix.shape}")
print(f"原始图像数据量: {image_matrix.size} 个像素值")

# 2. 对图像矩阵进行奇异值分解 (SVD)
# 这将分解为三个矩阵：U, Sigma, V^T
U, Sigma, VT = np.linalg.svd(image_matrix, full_matrices=False)
k = 33  # 选择保留的奇异值数量（压缩程度）

# 3. 使用前k个奇异值重建近似图像
# 仅使用最重要的部分信息重建图像
Sigma_k = np.diag(Sigma[:k])  # 取前k个奇异值形成对角矩阵
approx_image = U[:, :k] @ Sigma_k @ VT[:k, :]  # 重构近似图像矩阵

# 4. 显示原始图像与压缩后的图像
plt.figure(figsize=(12, 6))

# 原始图像
plt.subplot(1, 2, 1)
plt.imshow(image_matrix, cmap='gray')
plt.title('hello')
plt.axis('off')

# 压缩后的图像
plt.subplot(1, 2, 2)
plt.imshow(approx_image, cmap='gray')
plt.title(f'\nhello (use {k}/{len(Sigma)} value)')
plt.axis('off')

plt.tight_layout()
plt.show()

# 5. 计算压缩率
original_size = image_matrix.size
compressed_size = U[:, :k].size + Sigma_k.size + VT[:k, :].size

print(f"原始数据量: {original_size} 像素值")
print(f"压缩后存储需求: {compressed_size} 数值")
print(f"压缩率: {compressed_size / original_size:.2%}")
print(f"保留信息比例: {np.sum(Sigma[:k]) / np.sum(Sigma):.2%}")