import numpy as np

# 5.1 数据归一化：对于一个给定的随机一维数组（长度为20），将其归一化至[0,1]区间。归一化公式：(x-min)/(max-min)。
array = np.random.randint(1, 20, 20)
print(array)
mymin = array.min()
diff = array.max() - mymin
print(diff)
if diff == 0:
    print(np.full_like(array, 0.5, dtype=float))
else:
    print((array - mymin)/diff)

# 5.2 计算欧式距离：两个向量（一维数组）u和v，计算它们之间的欧式距离。公式：sqrt(sum((u-v)**2))。
u = np.random.randint(1, 10, 10)
v = np.random.randint(1, 10, 10)
print(u)
print(v)
# 方法1
print(np.sqrt(np.sum((u - v) ** 2)))
# 方法2
print(np.linalg.norm(u - v))

# 5.3 统计数组中每个元素出现的次数：给定一个随机整数数组（包含0-4的整数，长度为50），统计每个整数出现的次数（使用np.bincount）。
ri50 = np.random.randint(0, 5, 50)
print(ri50)
print(np.bincount(ri50))