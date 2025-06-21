import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. 加载彩色图片
image_path = 'C:\\Users\\Administrator\\Pictures\\color\\out24.png'  # 替换为你自己的图片路径
original_image = Image.open(image_path)  # 不转换为灰度图，保留RGB
image_matrix = np.array(original_image)  # 将图像转换为数值矩阵 (height, width, 3)

print(f"原始图像尺寸: {image_matrix.shape}")
print(f"原始图像数据量: {image_matrix.size} 个像素值")

# 2. 分离三个颜色通道
# 彩色图像有R、G、B三个通道
red_channel = image_matrix[:, :, 0]
green_channel = image_matrix[:, :, 1]
blue_channel = image_matrix[:, :, 2]

# 3. 为每个通道单独进行SVD
def svd_compress_channel(channel, k):
    """对单个颜色通道进行SVD压缩"""
    U, Sigma, VT = np.linalg.svd(channel, full_matrices=False)
    Sigma_k = np.diag(Sigma[:k])
    approx_channel = U[:, :k] @ Sigma_k @ VT[:k, :]
    return approx_channel, U.shape, Sigma.shape, VT.shape

k = 100  # 选择保留的奇异值数量（压缩程度）

# 对三个通道分别进行压缩
approx_red, r_U_shape, r_Sigma_shape, r_VT_shape = svd_compress_channel(red_channel, 100)
approx_green, g_U_shape, g_Sigma_shape, g_VT_shape = svd_compress_channel(green_channel, 100)
approx_blue, b_U_shape, b_Sigma_shape, b_VT_shape = svd_compress_channel(blue_channel, 10)

# 4. 合并压缩后的通道
# 将压缩后的三个通道组合成一个彩色图像
compressed_image = np.zeros_like(image_matrix)
compressed_image[:, :, 0] = np.clip(approx_red, 0, 255)  # 红色通道
compressed_image[:, :, 1] = np.clip(approx_green, 0, 255)  # 绿色通道
compressed_image[:, :, 2] = np.clip(approx_blue, 0, 255)  # 蓝色通道

# 5. 显示原始图像与压缩后的彩色图像
plt.figure(figsize=(12, 6))

# 原始彩色图像
plt.subplot(1, 2, 1)
plt.imshow(image_matrix)
plt.title('Original Color Image')
plt.axis('off')

# 压缩后的彩色图像
plt.subplot(1, 2, 2)
plt.imshow(compressed_image.astype(np.uint8))
plt.title(f'Compressed Color Image (k={k})')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. 计算压缩率
original_size = image_matrix.size  # 原始图像数据量（所有通道）
h, w, c = image_matrix.shape

# 压缩后数据量（所有通道的总和）
compressed_size = (
    (k * (r_U_shape[0] + 1 + r_VT_shape[1])) +  # 红色通道 (U[:, :k], Sigma_k, VT[:k, :])
    (k * (g_U_shape[0] + 1 + g_VT_shape[1])) +   # 绿色通道
    (k * (b_U_shape[0] + 1 + b_VT_shape[1]))     # 蓝色通道
)

print(f"\n原始图像尺寸: {h}×{w}×3 = {h*w*3} 像素值")
print(f"压缩后存储需求: {compressed_size} 个数值")
print(f"压缩率: {compressed_size / original_size:.2%}")
print(f"信息保留情况: 每个通道保留 {k} 个奇异值 (共 {image_matrix.shape[0]} 个)")