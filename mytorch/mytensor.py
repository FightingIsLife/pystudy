import torch
import numpy as np
import torch.nn as nn

# 从Java视角理解（对比原生数组）
# int[] javaArray = {1,2,3};
torch_tensor = torch.tensor([1, 2, 3])  # 一维张量（向量）

# 关键操作（无需数学推导）：
x = torch.randn(3, 4)  # 模拟Java: new float[3][4]
y = x[:, 1:3]  # 切片操作 → Python语法直接迁移
print(x)
print(y)
z = torch.cat([x, x], dim=0)  # 数组拼接 → System.arraycopy进阶版

print(z)

zx = torch.cat([x, x], dim=1)

print(zx)

# 场景：已知函数 y = x²，求 x=3 处的导数
x = torch.tensor(3.0, requires_grad=True)
print(x)
y = x ** 2
print(y.backward())  # 自动计算梯度
print(x.grad)  # 输出: tensor(4.0) → 即 2x 在x=2的值



# Java开发者类比：类似实现一个简单的Spring Boot Controller
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # 全连接层 → 类似定义DTO字段
        self.fc2 = nn.Linear(128, 10)  # 输出层 → 类似Controller返回结构

    def forward(self, x):  # 类似Spring MVC的请求处理流程
        x = torch.relu(self.fc1(x))  # 激活函数 → 业务逻辑处理
        x = self.fc2(x)  # 输出层处理
        return x


model = Net()
print(model)  # 查看结构（会打印各层维度）
