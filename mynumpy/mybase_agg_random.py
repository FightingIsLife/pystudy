import numpy as np

# 创建一个1到100的随机一维数组（长度为50），使用np.random.randint。
r50 = np.random.randint(1, 101, 50)
print(r50)

# 计算数组的和、最小值、最大值、均值、标准差、方差。
print(r50.sum())
print(r50.min())
print(r50.max())
print(r50.mean())
print(r50.std())
print(r50.var())

# 计算数组的最小值和最大值的索引位置。
print(r50.argmin())
print(r50.argmax())

# 创建一个4x4的随机整数数组（范围0-99），按行求和、按列求最大值、求整个数组的最小值。
arr44 = np.random.randint(0, 100, 16).reshape(4, 4)
print(arr44)
print(arr44.sum(axis=1))
print(arr44.max(axis=0))
print(arr44.min())

# 生成10个[0,1)之间均匀分布的随机小数。
rf10 = np.random.rand(10)
print(rf10)

# 生成100个标准正态分布（均值为0，标准差为1）的随机数。
normal_random = np.random.randn(100)
print(normal_random)
# 验证均值和标准差
mean_value = normal_random.mean()
std_value = normal_random.std()

print(f"均值: {mean_value:.4f} (接近0)")
print(f"标准差: {std_value:.4f} (接近1)")

# 从0到9随机抽取3个整数（不重复）。
rr3 = np.random.choice(10, 3, replace=False)
print(rr3)