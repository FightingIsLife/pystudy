import numpy as np

# 创建一个一维数组，包含10个0。
zero10 = np.zeros(10)
print(zero10)

# 创建一个3x3的二维数组，元素全为1。
matrix33Element1 = np.ones((3, 3))
print(matrix33Element1)

# 创建一个3x3的单位矩阵（对角线为1，其余为0）
eye3 = np.eye(3)
print(eye3)

# 创建一个长度为9的等间隔数组，从0开始到结束（包含0和结束值），步长为1（即0,1,2,...,8）。
array1 = np.arange(start=0, stop=9, step=1)
print(array1)
print(np.linspace(0, 8, 9))

# 创建一个3x3的二维数组，元素为0到8（按照行优先顺序）。
array33 = np.arange(9).reshape(3,3)
print(array33)

# 创建一个3x3的二维数组，元素为0到8（按照列优先顺序）。
col33 = np.arange(9).reshape(3,3, order='F')
print(col33)

# 转置
transposed = col33.T
print("\n转置数组:\n", transposed)

# 展平 (两种方法) 将转置后的数组展平为一维数组（使用两种方法：ravel和flatten，并理解两者的区别）
raveled = transposed.ravel()     # 视图(可能改变原数组)
flattened = transposed.flatten()  # 拷贝(安全独立)

print("\nravel()展平结果:", raveled)
print("flatten()展平结果:", flattened)

# 验证区别
raveled[0] = 100  # 修改ravel结果会影响原数组
print("\n修改ravel后影响转置数组:\n", transposed)
print("flatten结果不受影响:", flattened)

# 内存布局检查
print("\n== 内存布局验证 ==")
print(f"ravel() is view: {raveled.base is not None}")
print(f"flatten() is copy: {flattened.base is None}")